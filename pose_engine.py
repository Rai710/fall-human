"""Mesin deteksi jatuh berbasis VideoMAE (Hugging Face).

Pengganti pose_engine lama. Antarmuka (ROOT dan process_source) sengaja dibuat
sama persis dengan yang dipakai app.py, jadi UI tidak perlu diubah.

Catatan: VideoMAE mengklasifikasi seluruh adegan (bukan per orang), jadi tidak
ada kerangka tubuh maupun ID orang. Lihat keterangan di bagian stats di bawah.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForVideoClassification

ROOT = Path(__file__).resolve().parent

_THREADS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# --- Pengaturan -------------------------------------------------------------
MODEL_NAME = "yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect"
FALL_THRESHOLD = 0.75        # sama seperti script awal: confidence kelas jatuh harus > 75%
CONFIRM_WINDOWS = 2          # alarm baru bunyi jika N analisis berturut-turut bilang jatuh
ALERT_COOLDOWN_S = 10.0      # jeda minimal antar alarm (detik)
SAMPLES_PER_SECOND = 7.5     # 30 fps -> ambil 1 dari 4 frame (jarak sampling bawaan VideoMAE Kinetics)
INFER_EVERY_SAMPLES = 2      # analisis ulang tiap N sampel baru (30 fps: sekitar 0,27 detik)
SHORT_EDGE = 224             # sisi terpendek gambar yang masuk ke model

# Warna BGR untuk overlay
YELLOW, GREEN, RED, ORANGE = (0, 255, 255), (0, 200, 0), (0, 0, 255), (0, 165, 255)

_cache: dict = {}
_cache_lock = threading.Lock()


def load_model():
    """Muat model sekali saja; pemanggilan berikutnya langsung memakai cache."""
    with _cache_lock:
        if not _cache:
            print("Memuat model VideoMAE...")
            processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            model = AutoModelForVideoClassification.from_pretrained(MODEL_NAME)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            _cache.update(processor=processor, model=model, device=device)
        return _cache["processor"], _cache["model"], _cache["device"]


def fall_class_ids(model) -> set[int]:
    ids = {int(i) for i, name in model.config.id2label.items() if "fall" in str(name).lower()}
    if not ids:
        labels = ", ".join(str(v) for v in model.config.id2label.values())
        raise RuntimeError(f"Model tidak punya kelas 'fall'. Kelas yang ada: {labels}")
    return ids


def _predict(frames, processor, model, device, fall_ids):
    """16 frame PIL -> (label, confidence, apakah_jatuh)."""
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    idx = int(probs.argmax())
    conf = float(probs[idx])
    return model.config.id2label[idx], conf, (idx in fall_ids and conf > FALL_THRESHOLD)


def _to_pil(frame_bgr):
    """BGR OpenCV -> PIL RGB, dikecilkan dulu supaya buffer 16 frame tidak boros memori."""
    h, w = frame_bgr.shape[:2]
    scale = SHORT_EDGE / min(h, w)
    if scale < 1:
        frame_bgr = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)),
                               interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


class _AsyncPredictor:
    """Jalankan satu inferensi di thread terpisah agar loop webcam tidak beku."""

    def __init__(self, predict):
        self._predict = predict
        self._jobs = queue.Queue(maxsize=1)
        self._results = queue.Queue()
        self._stop = threading.Event()
        self._pending = False
        self._thread = threading.Thread(target=self._run, name="videomae-inference", daemon=True)
        self._thread.start()

    @property
    def busy(self):
        return self._pending

    def submit(self, frames, meta):
        if self._pending or self._stop.is_set():
            return False
        try:
            self._jobs.put_nowait((frames, meta))
        except queue.Full:
            return False
        self._pending = True
        return True

    def poll(self):
        try:
            result = self._results.get_nowait()
        except queue.Empty:
            return None
        self._pending = False
        return result

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            frames, meta = item
            started = time.monotonic()
            try:
                label, confidence, is_fall = self._predict(frames)
                self._results.put((label, confidence, is_fall,
                                   time.monotonic() - started, meta, None))
            except Exception as exc:  # noqa: BLE001
                self._results.put(("", 0.0, False,
                                   time.monotonic() - started, meta, exc))
        self._pending = False

    def close(self, timeout=0.5):
        self._stop.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)


def _draw_overlay(frame, text, color, alarm):
    h, w = frame.shape[:2]
    scale = max(0.6, min(h, w) / 720)
    thick = max(2, int(round(scale * 2)))
    if alarm:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), RED, max(4, int(6 * scale)))
    pos = (20, int(50 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, pos, font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, font, scale, color, thick, cv2.LINE_AA)


class _LatestFrame:
    """Baca webcam di thread terpisah dan selalu simpan gambar terbaru.

    Tanpa ini, saat model sedang menghitung, gambar webcam menumpuk di buffer
    dan tampilan jadi tertinggal beberapa detik.
    """

    def __init__(self, cap):
        self.cap = cap
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.alive = True
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.closed.is_set():
            ok, frame = self.cap.read()
            with self.cond:
                if not ok:
                    self.alive = False
                    self.cond.notify_all()
                    return
                self.frame = frame
                self.seq += 1
                self.cond.notify_all()

    def read(self, last_seq, timeout=0.5):
        """(ok, frame, seq). frame=None berarti belum ada gambar baru (timeout)."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.alive, timeout)
            if self.seq != last_seq:
                return True, self.frame, self.seq
            return self.alive, None, last_seq

    def close(self):
        self.closed.set()
        self.thread.join(timeout=2)


def process_source(source, save_video=True, stop=None, on_frame=None, realtime=True,
                   max_people=10, on_session=None):
    """Proses video (Path) atau webcam (int). Mengembalikan (folder_hasil, ringkasan).

    Parameter realtime dan max_people dipertahankan agar cocok dengan app.py,
    tetapi tidak dipakai: VideoMAE tidak melacak orang satu per satu.

    Isi stats yang dikirim ke on_frame (sesuai yang dibaca app.py):
      people / usable_poses -> selalu 0 (tidak ada deteksi per orang)
      new_alerts            -> track_id di sini adalah nomor kejadian jatuh (1, 2, 3, ...)
    """
    stop = stop or threading.Event()
    is_camera = isinstance(source, int)
    processor, model, device = load_model()
    fall_ids = fall_class_ids(model)
    num_frames = int(getattr(model.config, "num_frames", 16))

    cap = cv2.VideoCapture(source if is_camera else str(source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Tidak bisa membuka webcam nomor {source}." if is_camera
                           else f"Tidak bisa membuka video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not 1 <= fps <= 240:
        fps = 30.0
    total_frames = 0 if is_camera else max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    name = f"webcam{source}" if is_camera else Path(source).stem
    base = ROOT / "outputs" / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
    out_dir, suffix = base, 1
    while out_dir.exists():  # dua sesi dalam detik yang sama tidak boleh saling menimpa
        suffix += 1
        out_dir = base.with_name(f"{base.name}_{suffix}")
    out_dir.mkdir(parents=True)
    if on_session:
        on_session(out_dir)

    reader = _LatestFrame(cap) if is_camera else None
    pred_file = open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8")
    alert_file = open(out_dir / "alerts.csv", "w", newline="", encoding="utf-8")
    pred_csv, alert_csv = csv.writer(pred_file), csv.writer(alert_file)
    pred_csv.writerow(["frame", "timestamp_ms", "label", "confidence", "is_fall", "fall_streak"])
    alert_csv.writerow(["alert_id", "frame", "timestamp_ms", "label", "confidence"])
    writer = None
    warning = None

    window = collections.deque(maxlen=num_frames)
    interval = 1.0 / SAMPLES_PER_SECOND
    next_sample = 0.0
    since_infer = 0
    status_text, status_color, label = "Analyzing...", YELLOW, ""
    confidence = 0.0
    fall_streak = 0
    last_alert_ts = float("-inf")
    alerts_count = 0
    frame_no = 0
    seq = 0
    n_infer, infer_time = 0, 0.0
    fps_ema, last_tick = 0.0, None
    t0 = time.monotonic()

    async_predictor = None
    last_result_at = None
    if is_camera:
        async_predictor = _AsyncPredictor(
            lambda frames: _predict(frames, processor, model, device, fall_ids)
        )

    def apply_result(result_label, result_conf, result_is_fall, result_ts, result_frame,
                     elapsed, new_alerts):
        nonlocal label, confidence, fall_streak, last_alert_ts, alerts_count
        nonlocal n_infer, infer_time, status_text, status_color, last_result_at

        label = result_label
        confidence = result_conf
        fall_streak = fall_streak + 1 if result_is_fall else 0
        n_infer += 1
        infer_time += elapsed
        last_result_at = time.monotonic()

        if result_is_fall:
            status_text = f"WARNING: FALL DETECTED ({result_conf * 100:.1f}%)"
            status_color = RED
        elif "fall" in result_label.lower():
            status_text = f"Checking - {result_label} ({result_conf * 100:.1f}%)"
            status_color = ORANGE
        else:
            status_text = f"Normal - {result_label} ({result_conf * 100:.1f}%)"
            status_color = GREEN

        pred_csv.writerow([result_frame, int(result_ts * 1000), result_label,
                           f"{result_conf:.4f}", int(result_is_fall), fall_streak])
        pred_file.flush()

        # Alarm hanya saat streak baru mencapai batas (bukan terus-menerus
        # selama orang masih terbaring) dan di luar masa jeda.
        if fall_streak == CONFIRM_WINDOWS and result_ts - last_alert_ts >= ALERT_COOLDOWN_S:
            alerts_count += 1
            last_alert_ts = result_ts
            new_alerts.append({"track_id": alerts_count, "timestamp_ms": int(result_ts * 1000),
                               "label": result_label, "confidence": result_conf})
            alert_csv.writerow([alerts_count, result_frame, int(result_ts * 1000),
                                result_label, f"{result_conf:.4f}"])
            alert_file.flush()

    try:
        while not stop.is_set():
            if reader:
                ok, frame, seq = reader.read(seq)
                if not ok:
                    warning = "Webcam berhenti mengirim gambar."
                    break
                if frame is None:
                    continue
                ts = time.monotonic() - t0
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = frame_no / fps
            frame_no += 1
            new_alerts = []

            if async_predictor is not None:
                result = async_predictor.poll()
                if result is not None:
                    result_label, result_conf, result_is_fall, elapsed, meta, error = result
                    if error is not None:
                        warning = f"Inference gagal: {error}"
                        stop.set()
                        break
                    apply_result(result_label, result_conf, result_is_fall,
                                 meta[1], meta[0], elapsed, new_alerts)

            if ts + 1e-6 >= next_sample:
                next_sample += interval
                if next_sample < ts:
                    next_sample = ts + interval
                window.append(_to_pil(frame))
                since_infer += 1
                if len(window) == num_frames and since_infer >= INFER_EVERY_SAMPLES:
                    if async_predictor is not None:
                        if not async_predictor.busy:
                            if async_predictor.submit(list(window), (frame_no, ts)):
                                since_infer = 0
                    else:
                        since_infer = 0
                        started = time.monotonic()
                        result_label, result_conf, result_is_fall = _predict(
                            window, processor, model, device, fall_ids)
                        apply_result(result_label, result_conf, result_is_fall,
                                     ts, frame_no, time.monotonic() - started, new_alerts)

            _draw_overlay(frame, status_text, status_color, fall_streak >= CONFIRM_WINDOWS)

            if save_video and not is_camera:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        warning = "Video hasil tidak bisa disimpan (codec mp4v tidak tersedia)."
                if writer.isOpened():
                    writer.write(frame)

            now = time.monotonic()
            if last_tick is not None and now > last_tick:
                inst = 1.0 / (now - last_tick)
                fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            last_tick = now

            if on_frame:
                on_frame(frame, {
                    "people": 0,
                    "usable_poses": 0,
                    "frame": frame_no,
                    "timestamp_ms": int(ts * 1000),
                    "alerts_count": alerts_count,
                    "inference_fps": fps_ema,
                    "analyses": n_infer,
                    "device": str(device),
                    "total_frames": total_frames,
                    "new_alerts": new_alerts,
                    "label": label,
                    "confidence": confidence,
                    "fall_streak": fall_streak,
                    "is_fall": fall_streak >= CONFIRM_WINDOWS,
                    "infer_busy": bool(async_predictor and async_predictor.busy),
                    "last_result_age_ms": (
                        int((now - last_result_at) * 1000) if last_result_at is not None else None
                    ),
                })
    finally:
        if async_predictor:
            async_predictor.close()
        if reader:
            reader.close()
        cap.release()
        if writer is not None:
            writer.release()
        pred_file.close()
        alert_file.close()

    if warning is None and device.type == "cpu" and n_infer:
        warning = f"Berjalan di CPU: rata-rata {infer_time / n_infer:.1f} dtk per analisis."

    summary = {
        "status": "stopped" if stop.is_set() else "completed",
        "source": f"webcam:{source}" if is_camera else str(Path(source).resolve()),
        "model": MODEL_NAME,
        "device": str(device),
        "frames_processed": frame_no,
        "analyses": n_infer,
        "alerts_count": alerts_count,
        "max_simultaneous_tracked": 0,
        "unique_track_ids": 0,
        "predictions_csv": "predictions.csv",
        "alerts_csv": "alerts.csv",
        "warning": warning,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return out_dir, summary