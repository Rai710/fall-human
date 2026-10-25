"""Camera-level VideoMAE with independent capture, preview, and inference.

Importing this module does not load/download a model or open a camera.
"""
from __future__ import annotations

import csv
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
from PIL import Image

from live_core import AlarmGate, ClipBuffer, LatestAnalyzer, LiveConfig

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect"
LOCAL_MODEL = ROOT / "models" / "videomae-fall"
MODEL_CACHE = ROOT / ".cache" / "huggingface" / "hub"
_models = {}
_model_lock = threading.Lock()


class VideoMAEPredictor:
    def __init__(self, model_path=None, local_only=False):
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        self.torch = torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        required = ("config.json", "preprocessor_config.json", "model.safetensors")
        complete_local = all((LOCAL_MODEL / name).is_file() for name in required)
        source = str(Path(model_path).resolve()) if model_path else str(LOCAL_MODEL) if complete_local else MODEL_NAME
        if model_path and not all((Path(model_path) / name).is_file() for name in required):
            raise FileNotFoundError("Folder model harus berisi config.json, preprocessor_config.json, dan model.safetensors.")
        self.source = source
        options = {"cache_dir": str(MODEL_CACHE), "local_files_only": local_only}
        try:
            self.processor = VideoMAEImageProcessor.from_pretrained(source, **options)
            self.model = VideoMAEForVideoClassification.from_pretrained(
                source, use_safetensors=True, **options).to(self.device).eval()
        except OSError as error:
            raise RuntimeError(
                "Model VideoMAE belum dapat dimuat. Untuk persiapan terpisah, jalankan "
                "download_model.py dengan Python proyek saat terhubung internet. "
                "Mode lokal saja memerlukan model yang sudah lengkap.") from error
        self.num_frames = int(self.model.config.num_frames)
        self.labels = {int(index): str(label) for index, label in self.model.config.id2label.items()}
        self.fall_id = next((index for index, label in self.labels.items() if label == "FallDown"), None)
        if self.fall_id is None:
            raise ValueError("Model yang dipilih tidak memiliki kelas FallDown.")
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def predict(self, frames):
        inputs = self.processor(list(frames), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            scores = self.model(**inputs).logits.softmax(dim=-1)[0].float().cpu().tolist()
        index = max(range(len(scores)), key=scores.__getitem__)
        return {"label": self.labels[index], "confidence": scores[index],
                "fall_score": scores[self.fall_id],
                "scores": {self.labels[i]: score for i, score in enumerate(scores)}}


def load_model(model_path=None, local_only=False):
    key = (str(Path(model_path).resolve()) if model_path else "default", bool(local_only))
    with _model_lock:
        if key not in _models:
            _models[key] = VideoMAEPredictor(model_path, local_only)
        return _models[key]


def _image(frame):
    height, width = frame.shape[:2]
    scale = 224 / min(height, width)
    small = cv2.resize(frame, (round(width * scale), round(height * scale)),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


class _Display:
    def __init__(self, live, config):
        self.lock = threading.Lock()
        self.live, self.config = live, config
        self.value = {"label": "", "confidence": 0.0, "fall_score": 0.0,
                      "end_wall": None, "analysis_ms": None, "alarm": False, "stale": False}

    def update(self, **values):
        with self.lock:
            self.value.update(values)

    def snapshot(self):
        with self.lock:
            value = self.value.copy()
        end = value["end_wall"]
        value["result_age_ms"] = max(0, (time.monotonic() - end) * 1000) if end is not None else None
        if self.live and value["result_age_ms"] is not None:
            value["stale"] = value["result_age_ms"] > self.config.max_result_age * 1000
        return value


def _draw(frame, state, camera_name):
    height, width = frame.shape[:2]
    scale = max(0.4, min(0.8, min(height, width) / 650))
    if not state["label"]:
        text, color = "Mengumpulkan klip...", (0, 220, 255)
    elif state["stale"]:
        text, color = "HASIL TERLAMBAT - belum ada penilaian terbaru", (0, 190, 255)
    elif state["alarm"]:
        text, color = "ALARM KAMERA: DIDUGA JATUH", (70, 70, 255)
    else:
        text = f"{state['label']} | skor {state['confidence']:.0%}"
        color = (0, 190, 255) if state["label"] == "FallDown" else (140, 240, 140)
    if state["alarm"]:
        cv2.rectangle(frame, (2, 2), (width - 3, height - 3), (0, 0, 255), 4)
    lines = [camera_name[:45], text]
    if state["analysis_ms"] is not None:
        lines.append(f"Analisis {state['analysis_ms']/1000:.2f}s | usia hasil {state['result_age_ms']/1000:.2f}s")
    for number, line in enumerate(lines):
        position = (12, 24 + number * max(18, round(30 * scale)))
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return frame


class _Capture:
    def __init__(self, source, mode, config, analyzer, display, stop, out_dir,
                 save_video, num_frames, camera_name, frame_sink=None, renderer=None):
        self.source, self.mode, self.config = source, mode, config
        self.analyzer, self.display, self.stop = analyzer, display, stop
        self.out_dir, self.save_video, self.camera_name = out_dir, save_video, camera_name
        self.frame_sink = frame_sink
        self.renderer = renderer or _draw
        self.buffer = ClipBuffer(num_frames, config)
        self.condition = threading.Condition()
        self.done = threading.Event()
        self.error = None
        self.warning = None
        self.latest = None
        self.frames = self.total = self.recorded = 0
        self.fps = 0.0
        self.started_wall = None
        self.finished_wall = None
        self.max_capture_lag_ms = 0.0
        self.timestamp_fallbacks = 0
        self.annotated_video = None
        self.thread = threading.Thread(target=self._run, name="source-capture", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        capture, writer = cv2.VideoCapture(), None
        file_source = self.mode in ("replay", "offline")
        try:
            if self.mode == "stream":
                # Bound network reads; no automatic reconnect that could reuse
                # a clip from before a broken connection.
                okay = capture.open(str(self.source), cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
            else:
                okay = capture.open(str(self.source) if file_source else self.source)
            if not okay:
                raise RuntimeError("Sumber tidak dapat dibuka. Periksa file, webcam, atau alamat/akses CCTV.")
            raw_fps = capture.get(cv2.CAP_PROP_FPS)
            self.fps = raw_fps if math.isfinite(raw_fps) and 1 <= raw_fps <= 240 else 30.0
            raw_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self.total = max(0, int(raw_total)) if file_source and math.isfinite(raw_total) else 0
            self.started_wall = time.monotonic()
            previous_ts = -1.0
            raw_origin = None
            while not self.stop.is_set():
                okay, frame = capture.read()
                if self.stop.is_set():
                    break
                if not okay:
                    if not file_source:
                        raise RuntimeError("Aliran kamera terputus. Data sesi tersimpan; periksa koneksi lalu mulai sesi baru.")
                    break
                if file_source:
                    raw_ts = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if raw_origin is None:
                        raw_origin = raw_ts if math.isfinite(raw_ts) else 0.0
                    ts = raw_ts - raw_origin
                    if not math.isfinite(ts) or ts <= previous_ts:
                        ts = max(self.frames / self.fps, previous_ts + 1 / self.fps)
                        self.timestamp_fallbacks += 1
                    if self.mode == "replay" and self.stop.wait(max(0, self.started_wall + ts - time.monotonic())):
                        break
                else:
                    ts = time.monotonic() - self.started_wall
                acquired = time.monotonic()
                previous_ts = ts
                self.frames += 1
                self.max_capture_lag_ms = max(self.max_capture_lag_ms,
                    max(0, acquired - self.started_wall - ts) * 1000 if self.mode == "replay" else 0)
                with self.condition:
                    self.latest = (self.frames, ts, frame, acquired)
                    self.condition.notify_all()
                if self.frame_sink is not None:
                    self.frame_sink.offer((self.frames, ts, frame, acquired))
                    clip = None
                else:
                    clip = self.buffer.offer(ts, self.frames, acquired, lambda: _image(frame))
                if clip and self.analyzer.submit(clip) and self.mode == "offline":
                    # Keep offline annotations synchronized with delivery of
                    # the prediction, rather than racing the UI/log consumer.
                    self.analyzer.wait_for_delivery(clip.sequence)
                    if self.analyzer.error:
                        raise RuntimeError("Prediksi model gagal; lihat kesalahan analisis.") from self.analyzer.error
                if self.save_video and file_source:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(str(self.out_dir / "annotated.mp4"),
                                                cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                        if not writer.isOpened():
                            self.warning = "Video beranotasi gagal dibuat. CSV tetap disimpan."
                        else:
                            self.annotated_video = "annotated.mp4"
                    if writer.isOpened():
                        writer.write(self.renderer(frame.copy(), self.display.snapshot(), self.camera_name))
                        self.recorded += 1
            if file_source and self.total > self.frames + 1 and not self.stop.is_set():
                self.warning = "Decoder berhenti sebelum jumlah frame yang dilaporkan sumber."
        except Exception as error:
            self.error = error
        finally:
            capture.release()
            if writer:
                writer.release()
            self.finished_wall = time.monotonic()
            self.done.set()
            if self.frame_sink is not None:
                self.frame_sink.finish(cancel=self.stop.is_set() or self.error is not None)
            else:
                self.analyzer.finish(cancel=self.stop.is_set() or self.error is not None)
            with self.condition:
                self.condition.notify_all()

    def wait_latest(self, previous):
        with self.condition:
            self.condition.wait_for(lambda: self.latest is not None and self.latest[0] != previous
                                    or self.done.is_set(), timeout=0.04)
            return self.latest


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def process_source(source, save_video=True, stop=None, on_frame=None, realtime=True,
                   *, config=None, camera_name="Kamera 1", output_root=None,
                   model_path=None, local_only=False, on_event=None):
    """Run scene classification. No person counting or person IDs are produced.

    realtime=True paces video replay by its timestamps. Offline mode preserves
    every scheduled window. Webcam/network sources always use bounded live work.
    """
    config = config or LiveConfig()
    stop = stop if stop is not None else threading.Event()
    camera_name = camera_name.strip() or "Kamera 1"
    network = isinstance(source, str) and urlsplit(source).scheme.lower() in ("rtsp", "rtsps", "http", "https")
    mode = "webcam" if isinstance(source, int) else "stream" if network else "replay" if realtime else "offline"
    if mode in ("replay", "offline"):
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError("File video tidak ditemukan.")
    live = mode != "offline"
    # Never serialize URL credentials or query parameters into logs.
    source_description = f"webcam:{source}" if mode == "webcam" else "network_camera" if network else str(source.resolve())
    out_dir = Path(output_root or ROOT / "outputs") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_videomae_" + uuid4().hex[:6])
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 3, "backend": "VideoMAE scene classification", "mode": mode,
               "camera_name": camera_name, "source": source_description, "model": MODEL_NAME,
               "person_tracking": False, "config": config.to_dict(), "status": "processing",
               "started_at": datetime.now(timezone.utc).isoformat(), "alerts_count": 0,
               "prediction_csv": "predictions.csv", "alerts_csv": "alerts.csv", "warnings": []}
    fields = ["sequence", "clip_start_ms", "clip_end_ms", "end_frame", "label", "confidence",
              "fall_score", "analysis_ms", "result_age_ms", "stale", "fall_streak", "alert_id", "scores_json"]
    alert_fields = ["alert_id", "camera_name", "clip_start_ms", "clip_end_ms", "delivered_source_ms",
                    "label", "confidence", "fall_score", "analysis_ms", "result_age_ms"]
    capture = analyzer = None
    analysis_times, result_ages = [], []
    stale_count = preview_count = previous_frame = 0
    last_preview = None
    run_start = time.monotonic()

    def emit(kind, value):
        if on_event:
            on_event(kind, value)

    try:
        emit("status", "Menyiapkan VideoMAE dari penyimpanan lokal..." if local_only else
             "Menyiapkan VideoMAE. Jika diperlukan, model sekitar 345 MB akan diunduh...")
        predictor = load_model(model_path, local_only)
        summary.update(device=predictor.device, model_source=predictor.source,
                       model_revision=predictor.revision, num_frames=predictor.num_frames)
        if stop.is_set():
            summary["status"] = "stopped"
            return out_dir, summary
        display, gate = _Display(live, config), AlarmGate(config)
        with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as predictions_file, \
                (out_dir / "alerts.csv").open("w", newline="", encoding="utf-8") as alerts_file:
            predictions = csv.DictWriter(predictions_file, fieldnames=fields)
            alerts = csv.DictWriter(alerts_file, fieldnames=alert_fields)
            predictions.writeheader()
            alerts.writeheader()
            analyzer = LatestAnalyzer(predictor.predict, stop)
            capture = _Capture(source, mode, config, analyzer, display, stop, out_dir,
                               save_video, predictor.num_frames, camera_name)
            emit("status", f"Membuka {camera_name}; mode {mode}. Menunggu klip pertama...")
            capture.start()
            finishing_message = False
            while True:
                packet = capture.wait_latest(previous_frame)
                while True:
                    try:
                        clip, prediction, analysis_ms, finished = analyzer.results.get_nowait()
                    except queue.Empty:
                        break
                    # Include result delivery time, not just model forward time.
                    age_ms = max(0, (time.monotonic() - clip.end_wall) * 1000)
                    stale = live and age_ms > config.max_result_age * 1000
                    stale_count += int(stale)
                    analysis_times.append(analysis_ms)
                    result_ages.append(age_ms)
                    event = gate.update(clip, prediction, stale or stop.is_set())
                    display.update(label=prediction["label"], confidence=prediction["confidence"],
                                   fall_score=prediction["fall_score"], end_wall=clip.end_wall,
                                   analysis_ms=analysis_ms, alarm=gate.latched, stale=stale)
                    if event:
                        event.update(camera_name=camera_name, analysis_ms=analysis_ms,
                                     result_age_ms=age_ms,
                                     delivered_source_ms=round(packet[1] * 1000) if packet else round(clip.end_s * 1000))
                        alerts.writerow(event)
                        alerts_file.flush()
                        summary["alerts_count"] = gate.event_count
                        emit("alarm", event)
                    predictions.writerow(dict(sequence=clip.sequence,
                        clip_start_ms=round(clip.start_s * 1000), clip_end_ms=round(clip.end_s * 1000),
                        end_frame=clip.end_frame, label=prediction["label"],
                        confidence=prediction["confidence"], fall_score=prediction["fall_score"],
                        analysis_ms=round(analysis_ms, 2), result_age_ms=round(age_ms, 2),
                        stale=int(stale), fall_streak=gate.fall_streak,
                        alert_id=event["alert_id"] if event else "",
                        scores_json=json.dumps(prediction["scores"])))
                    predictions_file.flush()
                    analyzer.acknowledge(clip.sequence)
                if analyzer.error:
                    raise RuntimeError("Analisis VideoMAE gagal.") from analyzer.error
                if capture.error:
                    raise capture.error
                if packet:
                    frame_no, timestamp, frame, acquired = packet
                    state = display.snapshot()
                    # A latest-frame preview may skip displays, while the file
                    # recorder remains owned by capture and writes every frame.
                    if frame_no != previous_frame or capture.done.is_set():
                        previous_frame = frame_no
                        last_preview = _draw(frame.copy(), state, camera_name)
                        if on_frame:
                            elapsed = max(1e-6, (capture.finished_wall or time.monotonic()) - capture.started_wall)
                            on_frame(last_preview, {
                                "frame": frame_no, "timestamp_ms": round(timestamp * 1000),
                                "total_frames": capture.total, "mode": mode, "camera_name": camera_name,
                                "label": state["label"], "confidence": state["confidence"],
                                "fall_score": state["fall_score"], "stale": state["stale"],
                                "analysis_ms": state["analysis_ms"], "result_age_ms": state["result_age_ms"],
                                "capture_fps": capture.frames / elapsed,
                                "buffer_frames": len(capture.buffer.samples), "buffer_target": predictor.num_frames,
                                "analyses": analyzer.completed, "replaced_clips": analyzer.replaced,
                                "alerts_count": gate.event_count})
                            preview_count += 1
                if capture.done.is_set() and not analyzer.done.is_set() and not finishing_message:
                    emit("status", "Sumber selesai. Menunggu analisis yang masih berjalan dan menyimpan hasil...")
                    finishing_message = True
                if stop.is_set():
                    analyzer.finish(cancel=True)
                    if analyzer.done.is_set():
                        break
                if capture.done.is_set() and analyzer.done.is_set() and analyzer.results.empty():
                    break
                if capture.done.is_set():
                    analyzer.done.wait(0.04)
            summary["alerts_count"] = gate.event_count
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
        if analyzer:
            analyzer.finish(cancel=stop.is_set())
            analyzer.thread.join()
        if capture:
            capture.thread.join(timeout=4)
            if not capture.done.is_set():
                summary["warnings"].append("Pembaca sumber belum berhenti; backend kamera mungkin sedang menunggu I/O.")
            if capture.warning:
                summary["warnings"].append(capture.warning)
            summary.update(frames_processed=capture.frames, source_frame_count=capture.total,
                           source_fps=capture.fps, recorded_frames=capture.recorded,
                           annotated_video=capture.annotated_video,
                           max_capture_lag_ms=round(capture.max_capture_lag_ms, 2),
                           timestamp_fallbacks=capture.timestamp_fallbacks,
                           buffer_resets=capture.buffer.resets)
        if last_preview is not None:
            cv2.imwrite(str(out_dir / "preview.jpg"), last_preview)
        if not analysis_times and summary["status"] == "completed":
            summary["warnings"].append("Tidak ada klip lengkap yang dianalisis. Periksa durasi dan FPS sumber.")
        if stale_count:
            summary["warnings"].append(f"{stale_count} hasil melewati batas usia dan tidak dipakai mengonfirmasi alarm.")
        summary.update(analyses=len(analysis_times), stale_results=stale_count,
                       submitted_clips=analyzer.submitted if analyzer else 0,
                       replaced_clips=analyzer.replaced if analyzer else 0,
                       preview_updates=preview_count,
                       analysis_mean_ms=sum(analysis_times) / len(analysis_times) if analysis_times else None,
                       analysis_p95_ms=_percentile(analysis_times, 0.95),
                       result_age_mean_ms=sum(result_ages) / len(result_ages) if result_ages else None,
                       result_age_p95_ms=_percentile(result_ages, 0.95),
                       session_wall_seconds=time.monotonic() - run_start,
                       finished_at=datetime.now(timezone.utc).isoformat())
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir, summary
