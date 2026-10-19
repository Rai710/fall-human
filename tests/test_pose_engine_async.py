import threading
import time

import numpy as np

import pose_engine


def test_async_predictor_returns_result():
    started = threading.Event()
    release = threading.Event()

    def fake_predict(frames):
        started.set()
        release.wait(2)
        return "fall", 0.9, True

    predictor = pose_engine._AsyncPredictor(fake_predict)
    assert predictor.submit([1], ("meta",))
    assert started.wait(1)
    assert predictor.busy
    assert predictor.poll() is None

    release.set()
    deadline = time.monotonic() + 3
    result = None
    while time.monotonic() < deadline and result is None:
        result = predictor.poll()
        time.sleep(0.01)

    assert result is not None
    label, confidence, is_fall, elapsed, meta, error = result
    assert (label, confidence, is_fall, meta, error) == ("fall", 0.9, True, ("meta",), None)
    assert elapsed >= 0
    assert not predictor.busy
    predictor.close()


class _FakeCapture:
    def isOpened(self):
        return True

    def get(self, prop):
        if prop == pose_engine.cv2.CAP_PROP_FPS:
            return 30.0
        return 1.0

    def read(self):
        time.sleep(1 / 120)
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def release(self):
        pass


class _FakeConfig:
    num_frames = 16


class _FakeModel:
    config = _FakeConfig()


class _FakeDevice:
    type = "cpu"

    def __str__(self):
        return "cpu"


def test_camera_preview_keeps_running_during_inference(monkeypatch, tmp_path):
    monkeypatch.setattr(pose_engine, "ROOT", tmp_path)
    monkeypatch.setattr(pose_engine, "load_model", lambda: (None, _FakeModel(), _FakeDevice()))
    monkeypatch.setattr(pose_engine, "fall_class_ids", lambda model: {1})
    monkeypatch.setattr(pose_engine.cv2, "VideoCapture", lambda source: _FakeCapture())
    monkeypatch.setattr(pose_engine, "_to_pil", lambda frame: frame)
    monkeypatch.setattr(pose_engine, "_draw_overlay", lambda *args: None)

    def slow_predict(frames, processor, model, device, fall_ids):
        time.sleep(0.4)
        return "Walking", 0.8, False

    monkeypatch.setattr(pose_engine, "_predict", slow_predict)

    stop = threading.Event()
    timestamps = []
    analyses = []

    def on_frame(frame, stats):
        timestamps.append(time.monotonic())
        analyses.append(stats["analyses"])
        if stats["analyses"] >= 1:
            stop.set()

    watchdog = threading.Timer(8.0, stop.set)
    watchdog.start()
    try:
        pose_engine.process_source(0, save_video=False, stop=stop, on_frame=on_frame)
    finally:
        watchdog.cancel()

    assert analyses
    assert max(analyses) >= 1
    gaps = [later - earlier for earlier, later in zip(timestamps, timestamps[1:])]
    assert gaps
    assert max(gaps) < 0.2
