"""Local YOLO11 Pose + ByteTrack + RGB VideoMAE per tracked person.

Capture, tracking and action classification have independent workers. Pose
coordinates are not used to decide falls. Importing does not open a device.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from live_core import AlarmGate, ClipBuffer, LiveConfig
from person_core import PersonAnalyzer, PersonJob, TrackingConfig
from pose_engine import ROOT, MODEL_NAME, _Capture, _percentile, load_model

DETECTOR_PATH = ROOT / "models" / "yolo11n-pose.pt"
TRACKER_PATH = ROOT / "tracker.yaml"


def crop_clip(clip, margin):
    """Use one fixed square region for the whole clip, preserving fall motion.

    A moving/recentered crop on every frame could erase spatial motion. The
    region here is the union of the person's boxes across all real samples.
    Black padding avoids center-cropping away limbs at image boundaries.
    """
    boxes = np.asarray([sample[1] for sample in clip.frames], dtype=float)
    left, top = boxes[:, 0].min(), boxes[:, 1].min()
    right, bottom = boxes[:, 2].max(), boxes[:, 3].max()
    side = max(16, math.ceil(max(right - left, bottom - top) * (1 + 2 * margin)))
    x = math.floor((left + right - side) / 2)
    y = math.floor((top + bottom - side) / 2)
    images = []
    for frame, _ in clip.frames:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + side), min(h, y + side)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Crop orang berada di luar gambar.")
        patch = frame[y1:y2, x1:x2]
        patch = cv2.copyMakeBorder(patch, y1 - y, y + side - y2,
                                  x1 - x, x + side - x2, cv2.BORDER_CONSTANT)
        patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)))
    return replace(clip, frames=tuple(images))


class PersonDisplay:
    def __init__(self, live, config):
        self.live, self.config = live, config
        self.lock = threading.Lock()
        self.rows = []
        self.last = {}
        self.alarms = {}
        self.tracking_wall = None
        self.detector_ms = None
        self.detections = self.overflow = self.unassigned = 0

    def observations(self, rows, wall, detector_ms, detections, overflow, unassigned):
        with self.lock:
            previous = {(row["track_id"], row["segment_id"]): row for row in self.rows}
            for row in rows:
                old = previous.get((row["track_id"], row["segment_id"]), {})
                row.update({key: old[key] for key in ("result",) if key in old})
            self.rows = rows
            self.tracking_wall, self.detector_ms = wall, detector_ms
            self.detections, self.overflow, self.unassigned = detections, overflow, unassigned

    def is_current(self, job):
        with self.lock:
            exists = any((r["track_id"], r["segment_id"]) == job.key for r in self.rows)
            fresh = (not self.live or self.tracking_wall is not None
                     and time.monotonic() - self.tracking_wall <= 0.75)
            return exists and fresh

    def result(self, job, prediction, analysis_ms, stale, alarm):
        value = {**prediction, "analysis_ms": analysis_ms, "end_wall": job.clip.end_wall,
                 "stale": stale, "alarm": alarm, "track_id": job.track_id,
                 "segment_id": job.segment_id}
        with self.lock:
            self.last = value
            if alarm:
                self.alarms[job.key] = value
            else:
                self.alarms.pop(job.key, None)
            for row in self.rows:
                if (row["track_id"], row["segment_id"]) == job.key:
                    row["result"] = value

    def snapshot(self):
        with self.lock:
            rows = [dict(row) for row in self.rows]
            last = dict(self.last)
            tracking_wall = self.tracking_wall
            value = dict(rows=rows, last=last, tracking_wall=tracking_wall,
                         detector_ms=self.detector_ms, detections=self.detections,
                         overflow=self.overflow, unassigned=self.unassigned,
                         alarms=list(self.alarms))
        now = time.monotonic()
        for row in rows:
            if "result" in row:
                row["result"] = dict(row["result"])
                result = row["result"]
                result["result_age_ms"] = max(0, now - result["end_wall"]) * 1000
                result["stale"] |= self.live and result["result_age_ms"] > self.config.max_result_age * 1000
        if last:
            last["result_age_ms"] = max(0, now - last["end_wall"]) * 1000
            last["stale"] |= self.live and last["result_age_ms"] > self.config.max_result_age * 1000
        value["tracking_stale"] = (self.live and tracking_wall is not None and now - tracking_wall > 0.75)
        return value


def draw_people(frame, state, camera_name):
    height, width = frame.shape[:2]
    scale = max(0.4, min(0.7, min(height, width) / 650))
    if state["alarms"]:
        identities = ", ".join(f"{identity}.{segment}" for identity, segment in state["alarms"])
        text = "DIDUGA JATUH | ID historis " + identities
        cv2.rectangle(frame, (2, 2), (width - 3, height - 3), (0, 0, 255), 3)
        cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, scale, (50, 50, 255), 1, cv2.LINE_AA)
    for row in state["rows"]:
        x1, y1, x2, y2 = (int(n) for n in row["bbox"])
        result = row.get("result")
        stale = state["tracking_stale"] or result and result["stale"]
        color = (0, 180, 255) if stale else (100, 230, 130)
        label = "mengumpulkan klip" if row["analysis_enabled"] else "DI LUAR BATAS ANALISIS"
        if result:
            label = f"{result['label']} {result['confidence']:.0%}"
            if result["alarm"]:
                color = (40, 40, 255)
                label = "DIDUGA JATUH | " + label
            if stale:
                label += " | HASIL LAMA"
        if state["tracking_stale"]:
            label = "TRACKING LAMA"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"ID {row['track_id']}.{row['segment_id']} | {label}"
        position = (max(3, x1), max(22, y1 - 8))
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    text = f"{camera_name[:35]} | terlihat {state['detections']} | ID {len(state['rows'])}"
    cv2.putText(frame, text, (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (230, 230, 230), 1, cv2.LINE_AA)
    return frame


class TrackingWorker:
    """One latest raw frame, independent of capture and VideoMAE workers."""
    def __init__(self, detector, device, analyzer, display, stop, live, config,
                 tracking, num_frames, out_dir):
        self.detector, self.device = detector, device
        self.analyzer, self.display, self.stop = analyzer, display, stop
        self.live, self.config, self.tracking = live, config, tracking
        self.num_frames, self.out_dir = num_frames, out_dir
        self.condition = threading.Condition()
        self.pending = None
        self.closing = False
        self.done = threading.Event()
        self.error = None
        self.next_detection = 0.0
        self.last_processed = 0
        self.replaced = self.processed = self.peak_people = self.overflow_frames = 0
        self.unassigned_frames = self.reset_count = 0
        self.timings, self.ages = [], []
        self.states, self.segments = {}, {}
        self.unique_ids = set()
        self.thread = threading.Thread(target=self._run, name="yolo-bytetrack", daemon=True)

    def start(self):
        self.thread.start()

    def offer(self, packet):
        with self.condition:
            if self.closing or self.stop.is_set():
                return
            if packet[1] + 1e-7 < self.next_detection:
                return
            self.next_detection = packet[1] + 1 / self.tracking.detection_fps
            self.replaced += int(self.pending is not None)
            self.pending = packet
            self.condition.notify_all()
            if not self.live:
                self.condition.wait_for(lambda: self.last_processed >= packet[0]
                                        or self.error or self.stop.is_set())

    def finish(self, cancel=False):
        with self.condition:
            self.closing = True
            if cancel:
                self.pending = None
            self.condition.notify_all()

    def _run(self):
        try:
            with (self.out_dir / "tracking.csv").open("w", newline="", encoding="utf-8") as log:
                fields = ("frame", "timestamp_ms", "track_id", "segment_id", "confidence",
                          "x1", "y1", "x2", "y2", "analysis_enabled", "detector_ms", "tracking_age_ms")
                writer = csv.DictWriter(log, fieldnames=fields)
                writer.writeheader()
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.pending is not None
                                                or self.closing or self.stop.is_set(), timeout=0.2)
                        if self.stop.is_set():
                            break
                        if self.pending is None:
                            if self.closing:
                                break
                            continue
                        packet, self.pending = self.pending, None
                    frame_no, ts, frame, acquired = packet
                    began = time.monotonic()
                    result = self.detector.track(
                        frame, persist=True, tracker=str(TRACKER_PATH), classes=[0],
                        conf=0.1, imgsz=self.tracking.image_size, device=self.device,
                        max_det=64, verbose=False)[0]
                    # Include transfers and conversion in detector+tracker time.
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else []
                    detector_ms = (time.monotonic() - began) * 1000
                    tracking_age_ms = (time.monotonic() - acquired) * 1000
                    self.timings.append(detector_ms)
                    self.ages.append(tracking_age_ms)
                    self.processed += 1
                    self.peak_people = max(self.peak_people, len(xyxy))
                    self.unassigned_frames += int(len(ids) < len(xyxy))
                    people = {int(identity): (tuple(map(float, box)), float(score))
                              for identity, box, score in zip(ids, xyxy, scores)
                              if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]}
                    # Keep a short gap without inventing observations. Split
                    # only after the maximum allowed gap, even if an ID returns.
                    for identity, state in list(self.states.items()):
                        if ts - state["last_ts"] > self.config.max_sample_gap:
                            if state["active"]:
                                state["active"] = False
                                self.reset_count += 1
                            if ts - state["last_ts"] > 30:
                                del self.states[identity]
                    # Keep current admitted IDs before admitting newcomers.
                    admitted = [identity for identity in people if identity in self.states and self.states[identity]["active"]]
                    admitted += [identity for identity in people if identity not in admitted]
                    admitted = set(admitted[:self.tracking.max_people])
                    overflow = len(people) - len(admitted)
                    self.overflow_frames += int(overflow > 0)
                    h, w = frame.shape[:2]
                    factor = min(1.0, 960 / max(h, w))
                    sample_frame = cv2.resize(frame, (round(w * factor), round(h * factor))) if factor < 1 else frame
                    rows, jobs = [], []
                    for identity, (box, score) in people.items():
                        self.unique_ids.add(identity)
                        enabled = identity in admitted
                        if not enabled:
                            rows.append(dict(track_id=identity, segment_id=-1, bbox=box,
                                             confidence=score, analysis_enabled=False, buffer_frames=0))
                            continue
                        state = self.states.get(identity)
                        if state is None or not state["active"]:
                            segment = self.segments.get(identity, -1) + 1
                            self.segments[identity] = segment
                            state = dict(segment=segment, buffer=ClipBuffer(self.num_frames, self.config), active=True)
                            self.states[identity] = state
                        state["last_ts"] = ts
                        scaled_box = tuple(value * factor for value in box)
                        clip = state["buffer"].offer(ts, frame_no, acquired,
                            lambda box=scaled_box: (sample_frame, box))
                        rows.append(dict(track_id=identity, segment_id=state["segment"], bbox=box,
                                         confidence=score, analysis_enabled=True,
                                         buffer_frames=len(state["buffer"].samples)))
                        if clip:
                            jobs.append(PersonJob(identity, state["segment"],
                                                  crop_clip(clip, self.tracking.crop_margin), box))
                    self.display.observations(rows, acquired, detector_ms, len(xyxy), overflow,
                                              len(xyxy) - len(ids))
                    for row in rows:
                        box = row["bbox"]
                        writer.writerow(dict(frame=frame_no, timestamp_ms=round(ts * 1000),
                            track_id=row["track_id"], segment_id=row["segment_id"], confidence=row["confidence"],
                            x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                            analysis_enabled=int(row["analysis_enabled"]), detector_ms=round(detector_ms, 2),
                            tracking_age_ms=round(tracking_age_ms, 2)))
                    log.flush()
                    for job in jobs:
                        if self.analyzer.submit(job) and not self.live:
                            self.analyzer.wait_delivered(job)
                            if self.analyzer.error:
                                raise RuntimeError("Prediksi per orang gagal.") from self.analyzer.error
                    with self.condition:
                        self.last_processed = frame_no
                        self.condition.notify_all()
        except Exception as error:
            self.error = error
        finally:
            self.done.set()
            self.analyzer.finish(cancel=self.stop.is_set() or self.error is not None)
            with self.condition:
                self.condition.notify_all()


def process_people(source, save_video=True, stop=None, on_frame=None, realtime=True,
                   *, config=None, camera_name="Koridor 1", output_root=None,
                   model_path=None, local_only=True, on_event=None, tracking=None,
                   detector_path=None):
    config, tracking = config or LiveConfig(confirm_windows=1), tracking or TrackingConfig()
    stop = stop if stop is not None else threading.Event()
    camera_name = camera_name.strip() or "Koridor 1"
    network = isinstance(source, str) and urlsplit(source).scheme.lower() in ("rtsp", "rtsps", "http", "https")
    mode = "webcam" if isinstance(source, int) else "stream" if network else "replay" if realtime else "offline"
    live = mode != "offline"
    if mode in ("replay", "offline"):
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError("File video tidak ditemukan.")
    out_dir = Path(output_root or ROOT / "outputs") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_people_" + uuid4().hex[:6])
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = dict(schema_version=4, backend="YOLO11 Pose + ByteTrack + per-person VideoMAE",
        mode=mode, camera_name=camera_name, model=MODEL_NAME, person_tracking=True,
        source="network_camera" if network else str(source), status="processing",
        config=config.to_dict(), tracking_config=tracking.__dict__, alerts_count=0,
        started_at=datetime.now(timezone.utc).isoformat(), warnings=[])
    capture = tracker = analyzer = None
    inference_times, ages, queues = [], [], []
    stale_count = historical_count = preview_count = previous_frame = 0
    gates = {}
    last_preview = None
    run_start = time.monotonic()

    def emit(kind, payload):
        if on_event:
            on_event(kind, payload)

    try:
        detector_path = Path(detector_path or DETECTOR_PATH)
        if not detector_path.is_file():
            raise FileNotFoundError("YOLO lokal tidak ditemukan: models/yolo11n-pose.pt. Tidak ada unduhan otomatis.")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO
        emit("status", "Memuat YOLO + ByteTrack dan VideoMAE lokal...")
        predictor = load_model(model_path, local_only)
        detector = YOLO(str(detector_path), task="pose")
        device = "0" if predictor.device == "cuda" else "cpu"
        emit("status", "Menyiapkan detector/tracker sebelum sumber mulai dibaca...")
        detector.track(np.zeros((tracking.image_size, tracking.image_size, 3), np.uint8),
            persist=True, tracker=str(TRACKER_PATH), classes=[0], conf=0.1,
            imgsz=tracking.image_size, device=device, max_det=64, verbose=False)
        for internal_tracker in getattr(detector.predictor, "trackers", []):
            internal_tracker.reset()
        summary.update(device=predictor.device, detector_model=str(detector_path),
                       model_source=predictor.source, model_revision=predictor.revision,
                       num_frames=predictor.num_frames, startup_ms=(time.monotonic() - run_start) * 1000)
        if stop.is_set():
            summary["status"] = "stopped"
            return out_dir, summary
        display = PersonDisplay(live, config)
        prediction_fields = ("track_id", "segment_id", "sequence", "clip_start_ms", "clip_end_ms",
            "end_frame", "label", "confidence", "fall_score", "analysis_ms", "queue_ms", "result_age_ms",
            "stale", "identity_current", "skipped", "fall_streak", "alert_id", "scores_json")
        alert_fields = ("alert_id", "person_event_id", "track_id", "segment_id", "camera_name",
            "clip_start_ms", "clip_end_ms", "delivered_source_ms", "label", "confidence", "fall_score",
            "analysis_ms", "result_age_ms")
        with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as pred_log, \
                (out_dir / "alerts.csv").open("w", newline="", encoding="utf-8") as alert_log:
            predictions = csv.DictWriter(pred_log, fieldnames=prediction_fields)
            alerts = csv.DictWriter(alert_log, fieldnames=alert_fields)
            predictions.writeheader()
            alerts.writeheader()
            analyzer = PersonAnalyzer(predictor.predict, stop, tracking.max_people,
                                      config.max_result_age if live else None)
            tracker = TrackingWorker(detector, device,
                analyzer, display, stop, live, config, tracking, predictor.num_frames, out_dir)
            capture = _Capture(source, mode, config, analyzer, display, stop, out_dir,
                save_video, predictor.num_frames, camera_name, frame_sink=tracker, renderer=draw_people)
            tracker.start()
            capture.start()
            emit("status", f"Mode {mode}. Tampilan, tracking, dan prediksi per orang berjalan terpisah.")
            finishing_message = False
            while True:
                packet = capture.wait_latest(previous_frame)
                while True:
                    try:
                        job, prediction, duration, started, finished = analyzer.results.get_nowait()
                    except queue.Empty:
                        break
                    age = max(0, time.monotonic() - job.clip.end_wall) * 1000
                    wait_ms = max(0, started - job.clip.end_wall) * 1000
                    current = display.is_current(job)
                    stale = live and age > config.max_result_age * 1000
                    gate = gates.setdefault(job.key, AlarmGate(config))
                    skipped = prediction is None
                    event = None
                    if not skipped:
                        inference_times.append(duration)
                        ages.append(age)
                        queues.append(wait_ms)
                        stale_count += int(stale)
                        historical_count += int(not current)
                        # A real clip can show a fall just before detection is
                        # lost. Its fresh result remains valid historical
                        # evidence; never attach it to a new ID/segment.
                        event = gate.update(job.clip, prediction, stale or stop.is_set())
                        display.result(job, prediction, duration, stale, gate.latched)
                    else:
                        gate.fall_streak = gate.recovery_streak = 0
                    if event:
                        summary["alerts_count"] += 1
                        event.update(person_event_id=event["alert_id"], alert_id=summary["alerts_count"],
                            track_id=job.track_id, segment_id=job.segment_id, camera_name=camera_name,
                            analysis_ms=round(duration, 2), result_age_ms=round(age, 2),
                            delivered_source_ms=round(packet[1] * 1000) if packet else round(job.clip.end_s * 1000))
                        alerts.writerow(event)
                        alert_log.flush()
                        emit("alarm", event)
                    prediction = prediction or dict(label="SKIPPED_EXPIRED", confidence=0, fall_score=0, scores={})
                    predictions.writerow(dict(track_id=job.track_id, segment_id=job.segment_id,
                        sequence=job.clip.sequence, clip_start_ms=round(job.clip.start_s * 1000),
                        clip_end_ms=round(job.clip.end_s * 1000), end_frame=job.clip.end_frame,
                        label=prediction["label"], confidence=prediction["confidence"], fall_score=prediction["fall_score"],
                        analysis_ms=round(duration, 2), queue_ms=round(wait_ms, 2), result_age_ms=round(age, 2),
                        stale=int(stale), identity_current=int(current), skipped=int(skipped),
                        fall_streak=gate.fall_streak, alert_id=event["alert_id"] if event else "",
                        scores_json=json.dumps(prediction["scores"])))
                    pred_log.flush()
                    if not live:
                        analyzer.acknowledge(job)
                if analyzer.error:
                    raise RuntimeError(f"Prediksi VideoMAE gagal: {analyzer.error}") from analyzer.error
                if tracker.error:
                    raise RuntimeError(f"Tracking gagal: {tracker.error}") from tracker.error
                if capture.error:
                    raise capture.error
                if packet:
                    frame_no, timestamp, frame, acquired = packet
                    state = display.snapshot()
                    previous_frame = frame_no
                    last_preview = draw_people(frame.copy(), state, camera_name)
                    if on_frame:
                        last = state["last"]
                        elapsed = max(1e-6, (capture.finished_wall or time.monotonic()) - capture.started_wall)
                        on_frame(last_preview, dict(frame=frame_no, timestamp_ms=round(timestamp * 1000),
                            total_frames=capture.total, mode=mode, camera_name=camera_name,
                            label=(f"ID {last['track_id']}.{last['segment_id']} {last['label']}" if last else ""),
                            confidence=last.get("confidence", 0), fall_score=last.get("fall_score", 0),
                            stale=last.get("stale", False), analysis_ms=last.get("analysis_ms"),
                            result_age_ms=last.get("result_age_ms"), capture_fps=capture.frames / elapsed,
                            detector_ms=state["detector_ms"], tracking_stale=state["tracking_stale"],
                            tracking_fps=tracker.processed / elapsed, detections=state["detections"],
                            tracked_people=len(state["rows"]), overflow_people=state["overflow"],
                            buffer_frames=max((r["buffer_frames"] for r in state["rows"]), default=0),
                            buffer_target=predictor.num_frames, analyses=analyzer.completed,
                            replaced_clips=analyzer.replaced, expired_clips=analyzer.expired,
                            pending_people=len(analyzer.pending), alerts_count=summary["alerts_count"]))
                        preview_count += 1
                if stop.is_set():
                    tracker.finish(cancel=True)
                    analyzer.finish(cancel=True)
                    if tracker.done.is_set() and analyzer.done.is_set():
                        break
                if capture.done.is_set() and tracker.done.is_set() and analyzer.done.is_set() and analyzer.results.empty():
                    break
                if capture.done.is_set() and not finishing_message:
                    emit("status", "Sumber selesai. Menunggu tracking/prediksi terakhir dan menyimpan hasil...")
                    finishing_message = True
                if capture.done.is_set():
                    time.sleep(0.02)
            summary["status"] = "stopped" if stop.is_set() else "completed"
    except KeyboardInterrupt:
        summary["status"] = "stopped"
        stop.set()
        raise
    except Exception as error:
        summary.update(status="failed", error=str(error))
        stop.set()
        raise
    finally:
        if tracker:
            tracker.finish(cancel=stop.is_set())
        if analyzer:
            analyzer.finish(cancel=stop.is_set())
        if tracker and tracker.thread.ident is not None:
            tracker.thread.join()
        if analyzer:
            analyzer.thread.join()
        if capture:
            capture.thread.join(timeout=4)
            if not capture.done.is_set():
                summary["warnings"].append("Pembaca sumber belum berhenti; backend menunggu I/O.")
            if capture.warning:
                summary["warnings"].append(capture.warning)
            summary.update(frames_processed=capture.frames, source_frame_count=capture.total,
                source_fps=capture.fps, recorded_frames=capture.recorded, annotated_video=capture.annotated_video,
                max_capture_lag_ms=capture.max_capture_lag_ms, timestamp_fallbacks=capture.timestamp_fallbacks)
            if mode == "replay":
                summary["replay_timing_valid"] = capture.max_capture_lag_ms <= 1000
                if not summary["replay_timing_valid"]:
                    summary["warnings"].append("Pembacaan replay tertinggal lebih dari 1 detik; pengukuran live sesi ini tidak stabil.")
        if tracker:
            summary.update(tracking_frames=tracker.processed, replaced_tracking_frames=tracker.replaced,
                detector_mean_ms=sum(tracker.timings) / len(tracker.timings) if tracker.timings else None,
                detector_p95_ms=_percentile(tracker.timings, 0.95),
                tracking_age_p95_ms=_percentile(tracker.ages, 0.95), peak_people=tracker.peak_people,
                unique_track_ids=len(tracker.unique_ids), clip_segment_resets=tracker.reset_count,
                overflow_frames=tracker.overflow_frames, unassigned_frames=tracker.unassigned_frames)
            if tracker.overflow_frames:
                summary["warnings"].append("Jumlah orang melewati batas analisis; lihat analysis_enabled di tracking.csv.")
        mean_ms = sum(inference_times) / len(inference_times) if inference_times else None
        summary.update(analyses=len(inference_times), stale_results=stale_count, historical_results=historical_count,
            analysis_mean_ms=mean_ms, analysis_p95_ms=_percentile(inference_times, 0.95),
            result_age_mean_ms=sum(ages) / len(ages) if ages else None,
            result_age_p95_ms=_percentile(ages, 0.95), queue_p95_ms=_percentile(queues, 0.95),
            submitted_clips=analyzer.submitted if analyzer else 0,
            replaced_clips=analyzer.replaced if analyzer else 0,
            expired_clips=analyzer.expired if analyzer else 0,
            evicted_clips=analyzer.evicted if analyzer else 0, preview_updates=preview_count,
            session_wall_seconds=time.monotonic() - run_start, finished_at=datetime.now(timezone.utc).isoformat())
        if mean_ms is not None and tracker:
            summary["estimated_serial_load_at_peak"] = mean_ms / 1000 * min(tracker.peak_people, tracking.max_people) / config.analysis_interval
            if summary["estimated_serial_load_at_peak"] > 1:
                summary["warnings"].append("Kapasitas prediksi serial di bawah target jeda analisis untuk jumlah orang puncak.")
        if stale_count:
            summary["warnings"].append(f"{stale_count} hasil terlambat tidak dipakai mengonfirmasi alarm.")
        if not inference_times and summary["status"] == "completed":
            summary["warnings"].append("Belum ada klip orang lengkap yang dianalisis; periksa durasi, deteksi, celah tracking, dan FPS.")
        if last_preview is not None:
            cv2.imwrite(str(out_dir / "preview.jpg"), last_preview)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir, summary
