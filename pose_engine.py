"""Multi-person YOLO11 Pose + ByteTrack with a conservative collapse rule."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from fall_rules import CollapseDetector

DEFAULT_MODEL = ROOT / "models" / "yolo11n-pose.pt"
TRACKER_PATH = ROOT / "tracker.yaml"
LANDMARK_NAMES = (
    "nose left_eye right_eye left_ear right_ear left_shoulder right_shoulder "
    "left_elbow right_elbow left_wrist right_wrist left_hip right_hip "
    "left_knee right_knee left_ankle right_ankle"
).split()
CONNECTIONS = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
               (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
               (13, 15), (12, 14), (14, 16))
CSV_FIELDS = ["frame_index", "timestamp_ms", "track_id", "segment_id", "observed",
              "track_status", "pose_usable", "landmark_index", "landmark_name",
              "x", "y", "confidence", "keypoint_valid", "bbox_x1", "bbox_y1",
              "bbox_x2", "bbox_y2", "detection_confidence"]
FRAME_FIELDS = ["frame_index", "timestamp_ms", "width", "height", "detections",
                "tracked_people", "unassigned_people", "usable_poses", "missing_tracks"]
DETECTION_FIELDS = ["frame_index", "timestamp_ms", "track_id", "bbox_x1", "bbox_y1",
                    "bbox_x2", "bbox_y2", "detection_confidence"]
ALERT_FIELDS = ["frame_index", "timestamp_ms", "track_id", "event", "evidence_source", "drop_body_ratio",
                "descent_ms", "horizontal_hold_ms", "evidence_ms", "bbox_aspect",
                "floor_rise_body_ratio"]


@dataclass
class Person:
    track_id: int | None
    bbox: tuple
    score: float
    points: np.ndarray


@dataclass
class TrackFrame:
    track_id: int
    segment_id: int
    person: Person | None
    pose_usable: bool

    @property
    def status(self):
        if self.person is None:
            return "temporarily_missing"
        return "tracked" if self.pose_usable else "insufficient_pose"


def valid_points(points, width, height, threshold=0.3):
    return (np.isfinite(points).all(axis=1) & (points[:, 2] >= threshold)
            & (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height))


class TrackRegistry:
    """Keep short missing observations explicit and split sequences at gaps.

    A ByteTrack ID is temporary. Segment IDs prevent training clips from silently
    crossing missing/low-quality poses; they do not repair identity switches.
    """
    def __init__(self, missing_grace_ms=1000):
        self.missing_grace_ms = missing_grace_ms
        self.active = {}
        self.segments = {}
        self.totals = {}

    def update(self, people, timestamp_ms, width, height):
        seen = set()
        records = []
        for person in people:
            track_id = person.track_id
            if track_id is None or track_id in seen:
                continue
            seen.add(track_id)
            mask = valid_points(person.points, width, height)
            usable = bool(mask.sum() >= 6 and mask[[5, 6, 11, 12]].sum() >= 2)
            state = self.active.get(track_id)
            if state is None or timestamp_ms - state["last_seen"] > self.missing_grace_ms:
                segment = self.segments.get(track_id, -1) + 1
                state = {"last_seen": timestamp_ms, "segment": segment, "gap": False}
                self.active[track_id] = state
            elif usable and state["gap"]:
                state["segment"] += 1
            self.segments[track_id] = state["segment"]
            state.update(last_seen=timestamp_ms, gap=not usable)
            records.append(TrackFrame(track_id, state["segment"], person, usable))
            total = self.totals.setdefault(track_id, {"first_seen_ms": timestamp_ms,
                                                     "observed_frames": 0, "usable_pose_frames": 0})
            total["last_seen_ms"] = timestamp_ms
            total["observed_frames"] += 1
            total["usable_pose_frames"] += int(usable)
            total["segments"] = state["segment"] + 1
        for track_id, state in list(self.active.items()):
            if track_id in seen:
                continue
            state["gap"] = True
            if timestamp_ms - state["last_seen"] <= self.missing_grace_ms:
                records.append(TrackFrame(track_id, state["segment"], None, False))
            else:
                del self.active[track_id]
        return sorted(records, key=lambda record: record.track_id)


def parse_result(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    if result.keypoints is None or result.keypoints.conf is None:
        raise RuntimeError("Model harus menghasilkan 17 titik tubuh dan skor confidence.")
    coordinates = result.keypoints.xy.cpu().numpy()
    confidence = result.keypoints.conf.cpu().numpy()
    if coordinates.shape[1:] != (17, 2):
        raise RuntimeError("Prototipe ini menggunakan susunan 17 titik COCO YOLO Pose.")
    ids = boxes.id.cpu().numpy() if boxes.is_track and boxes.id is not None else [None] * len(boxes)
    return [Person(int(identity) if identity is not None else None, tuple(box), float(score),
                   np.column_stack((xy, conf)))
            for identity, box, score, xy, conf in zip(
                ids, boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), coordinates, confidence)]


def landmark_rows(frame_index, timestamp_ms, record, width, height):
    person = record.person
    mask = valid_points(person.points, width, height) if person else np.zeros(17, dtype=bool)
    for index, name in enumerate(LANDMARK_NAMES):
        row = dict.fromkeys(CSV_FIELDS, "")
        row.update(frame_index=frame_index, timestamp_ms=timestamp_ms, track_id=record.track_id,
                   segment_id=record.segment_id, observed=int(person is not None),
                   track_status=record.status, pose_usable=int(record.pose_usable),
                   landmark_index=index, landmark_name=name, keypoint_valid=int(mask[index]))
        if person:
            x, y, score = map(float, person.points[index])
            if math.isfinite(score):
                row["confidence"] = score
            # Zero-confidence coordinates can be placeholders from the model.
            if score > 0 and math.isfinite(x) and math.isfinite(y):
                row.update(x=x / width, y=y / height)
            row.update(bbox_x1=float(person.bbox[0]) / width, bbox_y1=float(person.bbox[1]) / height,
                       bbox_x2=float(person.bbox[2]) / width, bbox_y2=float(person.bbox[3]) / height,
                       detection_confidence=person.score)
        yield row


PALETTE = ((80, 210, 255), (100, 240, 110), (240, 170, 80), (190, 100, 240),
           (80, 180, 250), (240, 210, 80), (220, 120, 150))


def draw_people(frame, people, fall_statuses=None, provisional_matches=None):
    height, width = frame.shape[:2]
    fall_statuses = fall_statuses or {}
    provisional_matches = provisional_matches or {}
    tracked = 0
    for person in people:
        identity = person.track_id
        tracked += identity is not None
        display_identity = identity if identity is not None else provisional_matches.get(id(person))
        fall_status = fall_statuses.get(display_identity, "normal")
        color = ((0, 0, 255) if fall_status == "alert" else
                 (0, 190, 255) if fall_status == "candidate" else
                 PALETTE[identity % len(PALETTE)] if identity is not None else (155, 155, 155))
        x1, y1, x2, y2 = (int(v) for v in person.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if fall_status == "alert" else 2)
        mask = valid_points(person.points, width, height)
        positions = {i: (int(p[0]), int(p[1])) for i, p in enumerate(person.points) if mask[i]}
        for start, end in CONNECTIONS:
            if start in positions and end in positions:
                cv2.line(frame, positions[start], positions[end], color, 2, cv2.LINE_AA)
        for position in positions.values():
            cv2.circle(frame, position, 3, color, -1, cv2.LINE_AA)
        label = (f"ID {display_identity}* | PERIKSA: DIDUGA JATUH" if fall_status == "alert" and identity is None else
                 f"ID {display_identity} | PERIKSA: DIDUGA JATUH" if fall_status == "alert" else
                 f"ID {display_identity}* | memeriksa gerakan" if fall_status == "candidate" and identity is None else
                 f"ID {display_identity} | memeriksa gerakan" if fall_status == "candidate" else
                 f"ID {identity} | {int(mask.sum())}/17 titik" if identity is not None else "Menunggu ID")
        text_y = min(height - 8, max(58, y1 - 8))
        cv2.putText(frame, label, (max(0, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (max(0, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
    active_alerts = [identity for identity, status in fall_statuses.items() if status == "alert"]
    cv2.rectangle(frame, (0, 0), (width, min(height, 40)), (0, 0, 145) if active_alerts else (30, 35, 28), -1)
    banner = (f"ALARM - periksa ID {', '.join(map(str, active_alerts))}" if active_alerts else
              f"{tracked} orang dengan ID | {len(people) - tracked} menunggu ID")
    cv2.putText(frame, banner,
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 245, 240), 1, cv2.LINE_AA)
    return frame


def process_source(source, output_root=ROOT / "outputs", model_path=DEFAULT_MODEL,
                   save_video=True, stop=None, on_frame=None, realtime=False,
                   max_frames=None, max_people=10, image_size=640, device="cpu"):
    stop = stop if stop is not None else threading.Event()
    if not isinstance(max_people, int) or not 1 <= max_people <= 100:
        raise ValueError("Batas orang harus antara 1 dan 100.")
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError("Model YOLO belum tersedia. Jalankan MULAI.cmd terlebih dahulu.")
    camera = isinstance(source, int)
    if not camera and not Path(source).is_file():
        raise FileNotFoundError(f"Video tidak ditemukan: {source}")
    capture = cv2.VideoCapture(source if camera else str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Sumber tidak bisa dibuka. Periksa video atau izin webcam Windows.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fallback_fps = not math.isfinite(fps) or not 1 <= fps <= 240
    if fallback_fps:
        fps = 30.0
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = max(0, int(count)) if not camera and math.isfinite(count) else 0
    session = Path(output_root) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_yolo_" + uuid4().hex[:6])
    try:
        (session / "tracks").mkdir(parents=True, exist_ok=False)
    except OSError:
        capture.release()
        raise
    registry = TrackRegistry()
    collapse = CollapseDetector()
    summary = {
        "schema_version": 2, "source": f"webcam:{source}" if camera else str(Path(source).resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "processing",
        "backend": "YOLO11-Pose + ByteTrack", "ultralytics_version": importlib.metadata.version("ultralytics"),
        "torch_version": torch.__version__, "model": model_path.name,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "tracker": "tracker.yaml", "tracker_sha256": hashlib.sha256(TRACKER_PATH.read_bytes()).hexdigest(),
        "device": device, "image_size": image_size, "source_fps": fps, "fps_fallback": fallback_fps,
        "source_frame_count": total_frames, "max_people": max_people, "keypoints_per_person": 17,
        "frames_processed": 0, "frames_with_pose": 0, "max_simultaneous_tracked": 0,
        "annotated_video": None, "csv": "landmarks.csv", "frame_csv": "frames.csv",
        "track_directory": "tracks", "detection_csv": "detections.csv",
        "fall_classifier": "rule_based_collapse_v1",
        "collapse_rule": collapse.settings(), "alerts_csv": "alerts.csv", "alerts_count": 0,
    }
    writer = None
    processed = detected_frames = 0
    inference_seconds = 0.0
    last_timestamp = -1
    first_frame_wall = None
    last_frame = None
    run_start = time.perf_counter()
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        # A fresh model/tracker per source prevents cross-session histories.
        model = YOLO(str(model_path), task="pose")
        with (session / "landmarks.csv").open("w", newline="", encoding="utf-8") as landmark_file, \
                (session / "frames.csv").open("w", newline="", encoding="utf-8") as frame_file, \
                (session / "detections.csv").open("w", newline="", encoding="utf-8") as detection_file, \
                (session / "alerts.csv").open("w", newline="", encoding="utf-8") as alert_file:
            landmark_writer = csv.DictWriter(landmark_file, fieldnames=CSV_FIELDS)
            landmark_writer.writeheader()
            frame_writer = csv.DictWriter(frame_file, fieldnames=FRAME_FIELDS)
            frame_writer.writeheader()
            detection_writer = csv.DictWriter(detection_file, fieldnames=DETECTION_FIELDS)
            detection_writer.writeheader()
            alert_writer = csv.DictWriter(alert_file, fieldnames=ALERT_FIELDS)
            alert_writer.writeheader()
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    if camera:
                        raise RuntimeError("Aliran webcam terputus. Data sebelumnya tetap disimpan.")
                    break
                if first_frame_wall is None:
                    first_frame_wall = time.perf_counter()
                if camera:
                    raw_timestamp = (time.perf_counter() - first_frame_wall) * 1000
                else:
                    raw_timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                    if not math.isfinite(raw_timestamp) or (processed > 0 and raw_timestamp <= last_timestamp):
                        raw_timestamp = processed * 1000 / fps
                timestamp = max(last_timestamp + 1, round(raw_timestamp))
                last_timestamp = timestamp
                height, width = frame.shape[:2]
                inference_start = time.perf_counter()
                result = model.track(frame, persist=True, tracker=str(TRACKER_PATH),
                                     conf=0.15, iou=0.5, imgsz=image_size,
                                     max_det=max_people, classes=[0], device=device, verbose=False)[0]
                people = parse_result(result)
                for person in people:
                    detection_writer.writerow(dict(
                        frame_index=processed, timestamp_ms=timestamp,
                        track_id=person.track_id if person.track_id is not None else "",
                        bbox_x1=float(person.bbox[0]) / width,
                        bbox_y1=float(person.bbox[1]) / height,
                        bbox_x2=float(person.bbox[2]) / width,
                        bbox_y2=float(person.bbox[3]) / height,
                        detection_confidence=person.score))
                inference_seconds += time.perf_counter() - inference_start
                records = registry.update(people, timestamp, width, height)
                fall_statuses, new_alerts = collapse.update(
                    records, timestamp, width, height, processed,
                    unassigned_people=[person for person in people if person.track_id is None])
                if new_alerts:
                    alert_writer.writerows(new_alerts)
                    alert_file.flush()
                    summary["alerts_count"] += len(new_alerts)
                tracked = sum(person.track_id is not None for person in people)
                usable = sum(record.pose_usable for record in records)
                missing = sum(record.person is None for record in records)
                for record in records:
                    rows = list(landmark_rows(processed, timestamp, record, width, height))
                    landmark_writer.writerows(rows)
                    track_path = session / "tracks" / f"track_{record.track_id:04d}.csv"
                    exists = track_path.exists()
                    with track_path.open("a", newline="", encoding="utf-8") as track_file:
                        per_track = csv.DictWriter(track_file, fieldnames=CSV_FIELDS)
                        if not exists:
                            per_track.writeheader()
                        per_track.writerows(rows)
                frame_writer.writerow(dict(frame_index=processed, timestamp_ms=timestamp, width=width,
                                           height=height, detections=len(people), tracked_people=tracked,
                                           unassigned_people=len(people) - tracked,
                                           usable_poses=usable, missing_tracks=missing))
                summary["max_simultaneous_tracked"] = max(summary["max_simultaneous_tracked"], tracked)
                detected_frames += usable > 0
                annotated = draw_people(frame, people, fall_statuses, collapse.provisional_matches)
                if save_video and not camera:
                    if writer is None:
                        writer = cv2.VideoWriter(str(session / "annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("Video keluaran gagal dibuat. Matikan opsi simpan video dan coba lagi.")
                        summary["annotated_video"] = "annotated.mp4"
                    writer.write(annotated)
                processed += 1
                last_frame = annotated
                if on_frame:
                    on_frame(annotated, dict(frame=processed, total_frames=total_frames,
                                             timestamp_ms=timestamp, people=tracked, usable_poses=usable,
                                             unassigned=len(people) - tracked, missing_tracks=missing,
                                             total_ids=len(registry.totals),
                                             active_alert_ids=[identity for identity, status in fall_statuses.items()
                                                               if status == "alert"],
                                             new_alerts=new_alerts, alerts_count=summary["alerts_count"],
                                             inference_fps=processed / max(inference_seconds, 1e-9)))
                if max_frames is not None and processed >= max_frames:
                    summary["status"] = "limited"
                    break
                if realtime and not camera:
                    stop.wait(max(0, processed / fps - (time.perf_counter() - first_frame_wall)))
            if not processed and not stop.is_set():
                raise RuntimeError("Tidak ada frame yang bisa dibaca dari video.")
        if summary["status"] == "processing":
            summary["status"] = "stopped" if stop.is_set() else "completed"
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if last_frame is not None:
            cv2.imwrite(str(session / "preview.jpg"), last_frame)
        summary.update(frames_processed=processed, frames_with_pose=detected_frames,
                       pose_detection_rate=detected_frames / processed if processed else 0,
                       last_timestamp_ms=last_timestamp if processed else None,
                       inference_fps=processed / max(inference_seconds, 1e-9),
                       processing_fps=processed / max(time.perf_counter() - run_start, 1e-9),
                       tracks=registry.totals, unique_track_ids=len(registry.totals),
                       finished_at=datetime.now(timezone.utc).isoformat())
        if not camera and summary["status"] == "completed" and total_frames > processed + 1:
            summary["warning"] = "Decoder berhenti lebih awal dari jumlah frame yang dilaporkan."
        (session / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return session, summary
