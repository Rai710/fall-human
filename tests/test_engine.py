"""Uji pose_engine tanpa model asli: torch/transformers dan model dibuat tiruan.

Jalankan dari folder proyek:  python tests/test_engine.py
Tidak mengunduh apa pun dan tidak menulis ke folder outputs (memakai folder sementara).
Jika mediapipe terpasang, sebagian uji memakai mediapipe asli (tanpa file model).
"""
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

# --- stub torch & transformers supaya import berhasil tanpa paket asli -------------
torch_stub = types.ModuleType("torch")
torch_stub.device = lambda name: SimpleNamespace(type=name)
torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch_stub
tf_stub = types.ModuleType("transformers")
tf_stub.AutoImageProcessor = object
tf_stub.AutoModelForVideoClassification = object
sys.modules["transformers"] = tf_stub

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import pose_engine as pe  # noqa: E402

WORK = Path(tempfile.mkdtemp(prefix="fall_engine_test_"))
pe.ROOT = WORK  # hasil sesi masuk folder sementara

LABELS = {0: "FallDown", 1: "LyingDown", 2: "SitDown", 3: "Sitting", 4: "StandUp", 5: "Standing", 6: "Walking"}
fake_model = SimpleNamespace(config=SimpleNamespace(id2label=LABELS, num_frames=16))
fake_device = SimpleNamespace(type="cpu")
pe.load_model = lambda: (None, fake_model, fake_device)
pe.USE_SKELETON = False  # uji lama: tanpa kerangka (kerangka diuji terpisah di bawah)


def make_video(path, seconds=12, fps=30, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert w.isOpened(), "tidak bisa membuat video uji"
    for i in range(seconds * fps):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.circle(img, (20 + (i * 3) % (size[0] - 40), 120), 15, (200, 200, 200), -1)
        w.write(img)
    w.release()


video, short = WORK / "sample.avi", WORK / "short.avi"
make_video(video, 12)
make_video(short, 5)

calls = {"n": 0, "window_len": []}


def scripted_predict(frames, *a, **k):
    """Analisis ke-10..14 = FallDown (95%), ke-20 = FallDown 60% (di bawah ambang)."""
    calls["n"] += 1
    calls["window_len"].append(len(frames))
    assert all(hasattr(f, "size") for f in frames)  # PIL image
    n = calls["n"]
    if 10 <= n <= 14:
        return "FallDown", 0.95, True
    if n == 20:
        return "FallDown", 0.60, False
    return "Walking", 0.97, False


def run(src, **kw):
    kw.setdefault("save_video", False)
    kw.setdefault("stop", threading.Event())
    return pe.process_source(src, **kw)


# ---------------------------------------------------------------- 1. video penuh (batch)
pe._predict = scripted_predict
got_alerts, stats_seen = [], []


def on_frame(frame, stats):
    for k in ("people", "usable_poses", "frame", "timestamp_ms", "alerts_count",
              "inference_fps", "total_frames", "new_alerts"):
        assert k in stats, f"stats kehilangan kunci {k}"
    got_alerts.extend(stats["new_alerts"])
    stats_seen.append(stats)


out_dir, summary = run(video, save_video=True, on_frame=on_frame, realtime=False)
print("1. video penuh:", {k: summary[k] for k in ("status", "frames_processed", "analyses", "alerts_count")})
assert summary["status"] == "completed" and summary["frames_processed"] == 360
assert summary["alerts_count"] == 1 and len(got_alerts) == 1 and got_alerts[0]["track_id"] == 1
assert got_alerts[0]["timestamp_ms"] >= 2000 and all(n == 16 for n in calls["window_len"])
for k in ("status", "alerts_count", "max_simultaneous_tracked", "unique_track_ids", "warning"):
    assert k in summary
assert {"predictions.csv", "alerts.csv", "summary.json", "annotated.mp4"} <= {p.name for p in out_dir.iterdir()}
cap = cv2.VideoCapture(str(out_dir / "annotated.mp4"))
n_out = 0
while cap.read()[0]:
    n_out += 1
cap.release()
assert n_out == 360

# ------------------------------------- 2. terbaring terus-menerus -> hanya satu alarm
pe._predict = lambda *a, **k: ("FallDown", 0.99, True)
alerts2 = []
out2, sum2 = run(video, on_frame=lambda f, s: alerts2.extend(s["new_alerts"]), realtime=False)
print("2. terus-menerus jatuh -> alarm:", sum2["alerts_count"])
assert sum2["alerts_count"] == 1 and not (out2 / "annotated.mp4").exists()

# ---------------------------------------------------------------- 3. tombol Hentikan
pe._predict = scripted_predict
calls["n"] = 0
stop = threading.Event()
out3, sum3 = run(video, save_video=True, stop=stop, realtime=False,
                 on_frame=lambda f, s: stop.set() if s["frame"] >= 50 else None)
print("3. dihentikan:", sum3["status"], sum3["frames_processed"])
assert sum3["status"] == "stopped" and sum3["frames_processed"] == 50 and (out3 / "summary.json").exists()

# ---------------------------------------------------------------- 4. galat sumber
for bad, text in ((WORK / "tidak_ada.mp4", "Tidak bisa membuka video"), (99, "Tidak bisa membuka webcam")):
    try:
        run(bad)
        raise SystemExit("seharusnya gagal")
    except RuntimeError as e:
        assert text in str(e)
print("4. sumber tidak ada -> galat jelas")


# ---------------------------------------------------------------- 5. pembaca webcam
class FakeCap:
    def __init__(self, n):
        self.i, self.n = 0, n

    def read(self):
        time.sleep(0.005)
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.full((10, 10, 3), self.i, np.uint8)


reader = pe._LatestFrame(FakeCap(40))
seq, seen = 0, []
while True:
    ok, frame, seq = reader.read(seq)
    if not ok:
        break
    if frame is not None:
        seen.append(int(frame[0, 0, 0]))
        time.sleep(0.02)  # konsumen lebih lambat -> harus melompat ke gambar terbaru
reader.close()
print("5. webcam: dibaca", len(seen), "dari 40")
assert seen == sorted(seen) and len(seen) < 40 and seen[-1] == 40

# ---------------------------------------------------------------- 6. label model
try:
    pe.fall_class_ids(SimpleNamespace(config=SimpleNamespace(id2label={0: "Walking", 1: "Sitting"})))
    raise SystemExit("seharusnya gagal")
except RuntimeError:
    pass
assert pe.fall_class_ids(fake_model) == {0}
print("6. label 'fall' dikenali")


# ---------------------------------------------------------------- 7. kehalusan
def slow_predict(frames, *a, **k):
    time.sleep(0.6)  # meniru CPU lambat
    return "Walking", 0.9, False


def max_gap(realtime):
    pe._predict = slow_predict
    stamps = []
    t = time.monotonic()
    run(short, on_frame=lambda f, s: stamps.append(time.monotonic()), realtime=realtime)
    return max(b - a for a, b in zip(stamps, stamps[1:])), time.monotonic() - t


old_gap, _ = max_gap(False)
new_gap, new_total = max_gap(True)
print(f"7. jeda terpanjang antar frame: sinkron {old_gap:.2f} dtk | thread+pacing {new_gap:.2f} dtk "
      f"(video 5 dtk diputar {new_total:.1f} dtk)")
assert old_gap >= 0.5 and new_gap < 0.25 and 4.7 <= new_total <= 7.5

# ---------------------------------------------------------------- 8. alarm lewat thread sampai ke UI
pe._predict = lambda *a, **k: (time.sleep(0.2), ("FallDown", 0.99, True))[1]
got8 = []
_, sum8 = run(short, on_frame=lambda f, s: got8.extend(s["new_alerts"]), realtime=True)
print("8. mode thread -> alarm summary:", sum8["alerts_count"], "| sampai UI:", len(got8))
assert sum8["alerts_count"] == len(got8) == 1 and sum8["analyses"] >= 2


# ---------------------------------------------------------------- 9. galat model
def broken(*a, **k):
    raise ValueError("input salah")


pe._predict = broken
try:
    run(short, realtime=True)
    raise SystemExit("seharusnya gagal")
except RuntimeError as e:
    assert "input salah" in str(e)
    print("9. galat model ->", e)
assert not [t for t in threading.enumerate() if t.name == "videomae-infer" and t.is_alive()]

# ---------------------------------------------------------------- 10. Hentikan saat model sibuk
pe._predict = slow_predict
stop10 = threading.Event()
t = time.monotonic()
_, sum10 = run(short, save_video=True, stop=stop10, realtime=True,
               on_frame=lambda f, s: stop10.set() if s["frame"] >= 60 else None)
print(f"10. Hentikan -> {sum10['status']} setelah {time.monotonic() - t:.1f} dtk")
assert sum10["status"] == "stopped"

# =============================================================== KERANGKA ================
pe.USE_SKELETON = True


class FakeSketcher:
    """Orang berbaring (horizontal) di y=0.8; titik ke-16 persis di tengah (x=0.5)."""
    instances = []

    def __init__(self, max_people):
        self.max_people, self.closed, self.calls = max_people, False, 0
        FakeSketcher.instances.append(self)

    def detect(self, small, ts_ms):
        self.calls += 1
        assert small.ndim == 3 and max(small.shape[:2]) <= pe.SKELETON_MAX_SIDE
        return [[(0.2 + 0.6 * i / 32, 0.8, 0.99) for i in range(33)]]

    def close(self):
        self.closed = True


def white_dot_present(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, f = cap.read()
    cap.release()
    assert ok
    h, w = f.shape[:2]
    patch = f[int(0.8 * h) - 3:int(0.8 * h) + 4, int(0.5 * w) - 3:int(0.5 * w) + 4]
    return bool((patch.min(axis=2) > 200).any())


pe._PoseSketcher = FakeSketcher

# ---- 11. kerangka tergambar & TIDAK memengaruhi keputusan (orang berbaring + model bilang Walking)
pe._predict = lambda *a, **k: ("Walking", 0.95, False)
for realtime, frame_idx in ((False, 5), (True, 100)):
    FakeSketcher.instances.clear()
    people_seen = []
    out, s = run(short, save_video=True, realtime=realtime, max_people=3,
                 on_frame=lambda f, st: people_seen.append((st["people"], st["usable_poses"])))
    inst = FakeSketcher.instances[0]
    print(f"11. kerangka (realtime={realtime}): skeleton={s['skeleton']} maks_orang={s['max_simultaneous_tracked']} "
          f"alarm={s['alerts_count']} panggilan_detect={inst.calls}")
    assert s["skeleton"] and s["max_simultaneous_tracked"] == 1 and inst.max_people == 3 and inst.closed
    assert s["alerts_count"] == 0, "kerangka tidak boleh ikut memicu alarm"
    assert (1, 1) in people_seen
    assert white_dot_present(out / "annotated.mp4", frame_idx), "kerangka tidak tergambar di video hasil"

# ---- 12. kerangka error di tengah jalan -> dimatikan, deteksi jatuh tetap jalan
class FlakySketcher(FakeSketcher):
    def detect(self, small, ts_ms):
        self.calls += 1
        if self.calls > 3:
            raise ValueError("mediapipe rusak")
        return super().detect(small, ts_ms)


pe._PoseSketcher = FlakySketcher
calls["n"] = 0
pe._predict = scripted_predict
for realtime in (False, True):
    calls["n"] = 0
    _, s = run(video if not realtime else short, realtime=realtime)
    print(f"12. kerangka error (realtime={realtime}): skeleton={s['skeleton']} | {s['warning']}")
    assert not s["skeleton"] and "Kerangka MediaPipe dimatikan" in s["warning"] and s["status"] == "completed"
    if not realtime:
        assert s["alerts_count"] == 1, "deteksi jatuh harus tetap jalan"


# ---- 13. kerangka tidak bisa dibuat (mediapipe/model hilang) -> sesi tetap jalan
def cannot_build(max_people):
    raise RuntimeError("model kerangka tidak bisa diunduh")


pe._PoseSketcher = cannot_build
calls["n"] = 0
_, s = run(video, realtime=False)
print("13. kerangka tak tersedia ->", s["warning"])
assert not s["skeleton"] and "nonaktif" in s["warning"] and s["alerts_count"] == 1

# ---- 14. mediapipe ASLI (jika terpasang): API dan jalur gagal-aman tanpa file model
try:
    import mediapipe as mp
except ImportError:
    print("14. mediapipe tidak terpasang di sini -> uji API asli dilewati (jalur 'tidak tersedia' diuji di no. 13)")
else:
    from mediapipe.tasks.python import vision
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((48, 64, 3), np.uint8))
    assert image.width == 64 and image.height == 48
    for attr in ("PoseLandmarker", "PoseLandmarkerOptions", "RunningMode"):
        assert hasattr(vision, attr), attr
    assert hasattr(vision.RunningMode, "VIDEO")
    # Bangun _PoseSketcher asli dengan file model yang sengaja tidak ada.
    src = (PROJECT / "pose_engine.py").read_text(encoding="utf-8")
    ns = {}
    real_cls_src = src[src.index("class _PoseSketcher"):src.index("class _LatestFrame")]
    ns.update(cv2=cv2, ensure_pose_model=lambda: WORK / "tidak_ada.task")
    exec(real_cls_src, ns)
    try:
        ns["_PoseSketcher"](2)
        raise SystemExit("seharusnya gagal tanpa file model")
    except SystemExit:
        raise
    except Exception as e:
        print(f"14. mediapipe {mp.__version__} asli: API cocok; tanpa file model -> {type(e).__name__} (ditangani engine)")
    # dan engine harus menanganinya tanpa menghentikan sesi
    pe._PoseSketcher = ns["_PoseSketcher"]
    calls["n"] = 0
    _, s = run(video, realtime=False)
    assert not s["skeleton"] and "nonaktif" in s["warning"] and s["alerts_count"] == 1
    print("    sesi tetap selesai dengan alarm normal:", s["warning"][:70], "...")

# ---- 15. gambar kerangka: titik di luar frame / visibilitas rendah tidak membuat galat
blank = np.zeros((100, 100, 3), np.uint8)
pe._draw_skeleton(blank, [[(-5.0, 9.0, 0.9)] * 33, [(0.5, 0.5, 0.05)] * 33, []], pe.GREEN)
assert blank[50, 50].sum() == 0, "titik dengan visibilitas rendah tidak boleh digambar"
print("15. gambar kerangka tahan titik di luar frame")

print("\nSEMUA UJI LULUS")
