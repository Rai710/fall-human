"""Mesin deteksi jatuh.

* VideoMAE (Hugging Face) adalah SATU-SATUNYA yang memutuskan jatuh atau tidak.
* MediaPipe hanya menggambar kerangka tubuh sebagai tampilan; hasilnya tidak
  ikut menentukan alarm. Jika MediaPipe atau model kerangkanya tidak tersedia,
  deteksi jatuh tetap berjalan tanpa kerangka.
* Di GPU NVIDIA (CUDA) model berjalan dengan fp16; tanpa GPU otomatis memakai CPU.

Antarmuka (ROOT dan process_source) dipakai oleh app.py.
VideoMAE menilai seluruh adegan, bukan per orang, jadi tidak ada ID orang:
"track_id" pada alarm adalah nomor kejadian jatuh (1, 2, 3, ...).
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

from download_model import MODEL_NAME, POSE_MODEL_PATH, ensure_pose_model  # noqa: F401

ROOT = Path(__file__).resolve().parent

# --- Pengaturan -------------------------------------------------------------
FALL_THRESHOLD = 0.75        # confidence kelas jatuh harus > 75%
CONFIRM_WINDOWS = 2          # alarm baru bunyi jika N analisis berturut-turut bilang jatuh
ALERT_COOLDOWN_S = 10.0      # jeda minimal antar alarm (detik)
SAMPLES_PER_SECOND = 7.5     # 30 fps -> ambil 1 dari 4 frame (jarak sampling bawaan VideoMAE Kinetics)
INFER_EVERY_SAMPLES = 2      # analisis ulang tiap N sampel baru (30 fps: sekitar 0,27 detik)
SHORT_EDGE = 224             # sisi terpendek gambar yang masuk ke model

USE_SKELETON = True          # False = matikan kerangka MediaPipe sepenuhnya
SKELETON_MAX_SIDE = 640      # gambar dikecilkan ke ukuran ini sebelum masuk MediaPipe
SKELETON_MAX_AGE_S = 0.6     # kerangka yang lebih tua dari ini tidak lagi digambar
MIN_VISIBILITY = 0.3         # titik tubuh dengan visibilitas di bawah ini tidak digambar
USABLE_VISIBILITY = 0.5      # rata-rata visibilitas untuk dihitung "pose cukup jelas"

# Warna BGR untuk overlay
YELLOW, GREEN, RED, ORANGE = (0, 255, 255), (0, 200, 0), (0, 0, 255), (0, 165, 255)

# Sambungan antar 33 titik tubuh MediaPipe (topologi standar).
POSE_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)

_cache: dict = {}
_cache_lock = threading.Lock()


def _warm_up(model, device):
    """Satu inferensi kosong agar analisis pertama tidak tersendat (terutama di GPU)."""
    try:
        cfg = model.config
        shape = (1, int(getattr(cfg, "num_frames", 16)), int(getattr(cfg, "num_channels", 3)),
                 int(cfg.image_size), int(cfg.image_size))
        with torch.inference_mode():
            model(pixel_values=torch.zeros(shape, dtype=model.dtype, device=device))
    except Exception as error:  # hanya mempercepat; tidak boleh menggagalkan aplikasi
        print(f"Pemanasan model dilewati: {error}")


def load_model():
    """Muat model sekali saja; pemanggilan berikutnya langsung memakai cache."""
    with _cache_lock:
        if not _cache:
            print("Memuat model VideoMAE...")
            processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
            model = AutoModelForVideoClassification.from_pretrained(MODEL_NAME)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            if device.type == "cuda":
                model.half()  # fp16: jauh lebih cepat di GPU, selisih akurasi kecil
                print(f"Model berjalan di GPU: {torch.cuda.get_device_name(0)} (fp16)")
            else:
                # Sisakan 1 core CPU agar tampilan dan pembacaan video tetap lancar.
                torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
                print("Model berjalan di CPU.")
            model.eval()
            _warm_up(model, device)
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
    pixel_values = processor(list(frames), return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device=device, dtype=model.dtype)
    with torch.inference_mode():
        probs = torch.softmax(model(pixel_values=pixel_values).logits.float(), dim=-1)[0]
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


def _small(frame_bgr):
    """Salinan kecil untuk MediaPipe. Selalu salinan, karena frame asli digambari setelahnya."""
    h, w = frame_bgr.shape[:2]
    scale = SKELETON_MAX_SIDE / max(h, w)
    if scale < 1:
        return cv2.resize(frame_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return frame_bgr.copy()


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


def _draw_skeleton(frame, poses, color):
    """Gambar kerangka. poses = daftar orang; tiap orang = daftar (x, y, visibilitas) 0..1."""
    h, w = frame.shape[:2]
    scale = max(0.6, min(h, w) / 720)
    line, dot = max(2, int(round(2 * scale))), max(3, int(round(3 * scale)))
    for pose in poses:
        points = [(int(x * w), int(y * h)) if vis >= MIN_VISIBILITY else None for x, y, vis in pose]
        for a, b in POSE_EDGES:
            if a < len(points) and b < len(points) and points[a] and points[b]:
                cv2.line(frame, points[a], points[b], color, line, cv2.LINE_AA)
        for point in points:
            if point:
                cv2.circle(frame, point, dot, (255, 255, 255), -1, cv2.LINE_AA)


class _PoseSketcher:
    """Pembungkus MediaPipe PoseLandmarker (Tasks API; mp.solutions sudah dihapus di versi baru).

    Dibuat di sini, dipakai di satu thread. Melempar galat saat dibuat jika mediapipe
    belum terpasang atau model kerangka tidak bisa diunduh.
    """

    def __init__(self, max_people):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = ensure_pose_model()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max(1, int(max_people)),
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, small_bgr, ts_ms):
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = max(int(ts_ms), self._last_ms + 1)  # mode VIDEO mewajibkan waktu yang terus naik
        self._last_ms = ts_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts_ms)
        return [[(p.x, p.y, getattr(p, "visibility", None) or 0.0) for p in pose]
                for pose in result.pose_landmarks]

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


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


class _InferenceWorker:
    """Jalankan pekerjaan berat di thread sendiri supaya loop video tidak pernah menunggu.

    Hanya satu pekerjaan berjalan pada satu waktu. Selama sibuk, pekerja lain yang
    mengirim akan ditolak (submit mengembalikan False) dan bisa mencoba lagi dengan
    data terbaru, jadi yang diproses selalu yang paling baru.
    """

    def __init__(self, fn, name="infer"):
        self.fn = fn
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        self.busy = threading.Event()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            payload, meta = job
            started = time.monotonic()
            try:
                out = self.fn(payload)
                self.results.put((meta, out, time.monotonic() - started, None))
            except Exception as error:  # diteruskan ke loop utama supaya tampil di aplikasi
                self.results.put((meta, None, 0.0, error))
            finally:
                self.busy.clear()

    def submit(self, payload, meta):
        if self.busy.is_set():
            return False
        self.busy.set()
        self.jobs.put((payload, meta))
        return True

    def collect(self):
        items = []
        while True:
            try:
                items.append(self.results.get_nowait())
            except queue.Empty:
                return items

    def close(self):
        self.jobs.put(None)
        self.thread.join(timeout=60)


def _device_label(device):
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    return str(device.type)


def process_source(source, save_video=True, stop=None, on_frame=None, realtime=True, max_people=10):
    """Proses video (Path) atau webcam (int). Mengembalikan (folder_hasil, ringkasan).

    realtime=True : video file diputar sesuai kecepatan aslinya; model dan kerangka jalan
                    di thread masing-masing sehingga tampilan tidak tersendat.
    realtime=False: proses secepat mungkin, semua dipanggil langsung (untuk batch/uji).
    max_people    : batas jumlah kerangka yang digambar per frame.

    Isi stats yang dikirim ke on_frame (sesuai yang dibaca app.py):
      people / usable_poses -> jumlah kerangka terdeteksi / yang cukup jelas (0 jika kerangka mati)
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

    reader = _LatestFrame(cap) if is_camera else None
    pred_file = open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8")
    alert_file = open(out_dir / "alerts.csv", "w", newline="", encoding="utf-8")
    pred_csv, alert_csv = csv.writer(pred_file), csv.writer(alert_file)
    pred_csv.writerow(["frame", "timestamp_ms", "label", "confidence", "is_fall", "fall_streak"])
    alert_csv.writerow(["alert_id", "frame", "timestamp_ms", "label", "confidence"])
    writer = None
    warnings: list[str] = []

    window = collections.deque(maxlen=num_frames)
    interval = 1.0 / SAMPLES_PER_SECOND
    next_sample = 0.0
    since_infer = 0
    status_text, status_color = "Analyzing...", YELLOW
    fall_streak = 0
    last_alert_ts = float("-inf")
    alerts_count = 0
    frame_no = 0
    seq = 0
    n_infer, infer_time = 0, 0.0
    fps_ema, last_tick = 0.0, None
    pending_alerts = []
    last_frame, last_ts = None, 0.0
    poses, poses_ts, max_seen = [], float("-inf"), 0

    # Video file: diputar sesuai kecepatan aslinya (tidak dikebut). Webcam sudah dibatasi
    # kamera. Model selalu jalan di thread sendiri, kecuali realtime=False (proses batch,
    # secepat mungkin dengan model dipanggil langsung supaya hasilnya berurutan).
    paced = realtime and not is_camera
    worker = (_InferenceWorker(lambda frames: _predict(frames, processor, model, device, fall_ids),
                               name="videomae-infer") if realtime else None)
    sketcher, pose_worker = None, None

    def apply_result(meta, outcome):
        """Terapkan hasil satu analisis ke status, CSV dan alarm."""
        nonlocal status_text, status_color, fall_streak, last_alert_ts, alerts_count
        window_frame, window_ts = meta
        label, conf, is_fall = outcome
        fall_streak = fall_streak + 1 if is_fall else 0
        if is_fall:
            status_text, status_color = f"WARNING: FALL DETECTED ({conf * 100:.1f}%)", RED
        elif "fall" in label.lower():
            status_text, status_color = f"Checking - {label} ({conf * 100:.1f}%)", ORANGE
        else:
            status_text, status_color = f"Normal - {label} ({conf * 100:.1f}%)", GREEN
        pred_csv.writerow([window_frame, int(window_ts * 1000), label, f"{conf:.4f}",
                           int(is_fall), fall_streak])
        pred_file.flush()
        # Alarm hanya saat streak baru mencapai batas (bukan terus-menerus selama orang
        # masih terbaring) dan di luar masa jeda. Waktu alarm = akhir jendela yang dianalisis.
        if fall_streak == CONFIRM_WINDOWS and window_ts - last_alert_ts >= ALERT_COOLDOWN_S:
            alerts_count += 1
            last_alert_ts = window_ts
            pending_alerts.append({"track_id": alerts_count, "timestamp_ms": int(window_ts * 1000),
                                   "label": label, "confidence": conf})
            alert_csv.writerow([alerts_count, window_frame, int(window_ts * 1000), label, f"{conf:.4f}"])
            alert_file.flush()

    def take_results():
        nonlocal n_infer, infer_time
        for meta, outcome, took, error in worker.collect():
            if error is not None:
                raise RuntimeError(f"Model gagal menganalisis video: {error}") from error
            n_infer += 1
            infer_time += took
            apply_result(meta, outcome)

    def disable_skeleton(error):
        """Kerangka hanya tampilan: bila gagal, matikan saja dan lanjutkan deteksi jatuh."""
        nonlocal sketcher, pose_worker, poses
        warnings.append(f"Kerangka MediaPipe dimatikan: {error}")
        if pose_worker:
            pose_worker.close()
        if sketcher:
            sketcher.close()
        sketcher, pose_worker, poses = None, None, []

    def update_skeleton(frame, ts):
        """Perbarui kerangka terbaru; di mode thread hanya mengirim frame jika pekerja bebas."""
        nonlocal poses, poses_ts
        try:
            if pose_worker is None:
                poses, poses_ts = sketcher.detect(_small(frame), ts * 1000), ts
                return
            for meta, outcome, _took, error in pose_worker.collect():
                if error is not None:
                    raise error
                poses, poses_ts = outcome, meta
            if not pose_worker.busy.is_set():
                pose_worker.submit((_small(frame), ts * 1000), ts)
        except Exception as error:
            disable_skeleton(error)

    def make_stats(frame_index, ts, shown):
        usable = sum(1 for pose in shown
                     if pose and sum(p[2] for p in pose) / len(pose) >= USABLE_VISIBILITY)
        stats = {
            "people": len(shown),
            "usable_poses": usable,
            "frame": frame_index,
            "timestamp_ms": int(ts * 1000),
            "alerts_count": alerts_count,
            "inference_fps": fps_ema,
            "total_frames": total_frames,
            "new_alerts": list(pending_alerts),
        }
        pending_alerts.clear()
        return stats

    t_start = time.monotonic()
    try:
        if USE_SKELETON:
            try:
                sketcher = _PoseSketcher(max_people)
                if realtime:
                    detector = sketcher
                    pose_worker = _InferenceWorker(lambda job: detector.detect(*job), name="mediapipe-pose")
            except Exception as error:
                warnings.append(f"Kerangka MediaPipe nonaktif: {error}")
                sketcher = pose_worker = None
        t_start = time.monotonic()  # mulai hitung waktu setelah semua model siap

        while not stop.is_set():
            if reader:
                ok, frame, seq = reader.read(seq)
                if not ok:
                    warnings.append("Webcam berhenti mengirim gambar.")
                    break
                if frame is None:
                    continue
                ts = time.monotonic() - t_start
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = frame_no / fps
                if paced:  # tahan sampai waktu tayang frame ini; langsung lanjut jika sudah terlambat
                    stop.wait(max(0.0, t_start + ts - time.monotonic()))
                    if stop.is_set():
                        break
            frame_no += 1

            if ts + 1e-6 >= next_sample:
                next_sample += interval
                if next_sample < ts:
                    next_sample = ts + interval
                window.append(_to_pil(frame))
                since_infer += 1

            if worker:
                take_results()
            if len(window) == num_frames and since_infer >= INFER_EVERY_SAMPLES:
                if worker is None:
                    started = time.monotonic()
                    outcome = _predict(list(window), processor, model, device, fall_ids)
                    infer_time += time.monotonic() - started
                    n_infer += 1
                    apply_result((frame_no, ts), outcome)
                    since_infer = 0
                elif worker.submit(list(window), (frame_no, ts)):
                    since_infer = 0

            if sketcher is not None:
                update_skeleton(frame, ts)  # harus sebelum frame digambari
            shown = poses if ts - poses_ts <= SKELETON_MAX_AGE_S else []
            max_seen = max(max_seen, len(shown))
            if shown:
                _draw_skeleton(frame, shown, status_color)
            _draw_overlay(frame, status_text, status_color, fall_streak >= CONFIRM_WINDOWS)

            if save_video and not is_camera:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        warnings.append("Video hasil tidak bisa disimpan (codec mp4v tidak tersedia).")
                if writer.isOpened():
                    writer.write(frame)

            now = time.monotonic()
            if last_tick is not None and now > last_tick:
                inst = 1.0 / (now - last_tick)
                fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            last_tick = now

            last_frame, last_ts = frame, ts
            if on_frame:
                on_frame(frame, make_stats(frame_no, ts, shown))

        if worker:  # tunggu analisis terakhir selesai supaya tercatat di CSV dan summary
            worker.close()
            take_results()
            if pending_alerts and on_frame and last_frame is not None:
                on_frame(last_frame, make_stats(frame_no, last_ts, []))
    finally:
        if worker:
            worker.close()
        if pose_worker:
            pose_worker.close()
        if sketcher:
            sketcher.close()
        if reader:
            reader.close()
        cap.release()
        if writer is not None:
            writer.release()
        pred_file.close()
        alert_file.close()

    if device.type == "cpu" and n_infer:
        warnings.append(f"Berjalan di CPU: rata-rata {infer_time / n_infer:.1f} dtk per analisis.")

    summary = {
        "status": "stopped" if stop.is_set() else "completed",
        "model": MODEL_NAME,
        "device": _device_label(device),
        "skeleton": sketcher is not None,
        "frames_processed": frame_no,
        "analyses": n_infer,
        "alerts_count": alerts_count,
        "max_simultaneous_tracked": max_seen,
        "unique_track_ids": 0,
        "warning": " | ".join(warnings) or None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    return out_dir, summary
