"""End-to-end: upload video -> API -> process_people -> history.

YOLO dan VideoMAE diganti objek palsu; yang diuji adalah rangkaian kabel
antara dashboard, engine per-orang, konfigurasi detektor, dan repository.
"""
import sys
import time
import types

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main
import backend.repository as repository
import person_engine
from detector_config import ResolvedDetector


class _Arr:
    def __init__(self, data):
        self.data = np.asarray(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FakeYOLO:
    instances = []

    def __init__(self, path, *args, **kwargs):
        self.path, self.kwargs = path, kwargs
        self.names = {0: "person", 1: "bukan-orang"}
        self.predictor = types.SimpleNamespace(trackers=[])
        self.track_calls = []
        FakeYOLO.instances.append(self)

    def track(self, frame, **kwargs):
        self.track_calls.append(kwargs)
        h, w = frame.shape[:2]
        box = [[w * 0.2, h * 0.2, w * 0.6, h * 0.9]]
        boxes = types.SimpleNamespace(xyxy=_Arr(box), conf=_Arr([0.9]), id=_Arr([1]))
        return [types.SimpleNamespace(boxes=boxes)]


class FakePredictor:
    device, source, revision, num_frames = "cpu", "fake", "test", 16

    def __init__(self):
        self.calls = 0

    def predict(self, frames):
        self.calls += 1
        if self.calls == 1:
            return {"label": "Standing", "confidence": 0.9, "fall_score": 0.02,
                    "scores": {"Standing": 0.9, "FallDown": 0.02}}
        return {"label": "FallDown", "confidence": 0.95, "fall_score": 0.95,
                "scores": {"Standing": 0.03, "FallDown": 0.95}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"w")
    FakeYOLO.instances.clear()
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(person_engine, "ROOT", tmp_path)
    monkeypatch.setattr(person_engine, "load_model", lambda *a, **k: FakePredictor())
    monkeypatch.setattr(person_engine, "resolve_detector",
                        lambda **k: ResolvedDetector(weights, "local", "best.pt", [0], 0.1))
    monkeypatch.setattr(repository, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(main, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "manager", main.SessionManager())
    return tmp_path


def make_video(path, seconds=6, fps=15):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for index in range(seconds * fps):
        frame = np.full((240, 320, 3), 40, np.uint8)
        cv2.rectangle(frame, (60 + index % 40, 50), (180, 220), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()


def wait_done(client, session_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/sessions/{session_id}").json()
        if data.get("state") in ("completed", "failed", "stopped"):
            return data
        time.sleep(0.3)
    raise AssertionError("sesi tidak selesai tepat waktu")


def test_config_endpoint_reports_detector():
    response = TestClient(main.app).get("/api/config")
    assert response.status_code == 200
    detector = response.json()["detector"]
    assert detector["source"] in ("pretrained", "local", "huggingface")
    assert "label" in detector and "ready" in detector


def test_upload_video_runs_people_pipeline_end_to_end(env):
    video = env / "uji.avi"
    make_video(video)
    client = TestClient(main.app)
    with video.open("rb") as handle:
        started = client.post("/api/sessions/start", data={"save_video": "false", "realtime": "false", "analysis_scope": "person"},
                              files={"file": ("kamera-lorong.avi", handle, "video/x-msvideo")})
    assert started.status_code == 200, started.text
    snapshot = started.json()
    assert snapshot["name"] == "kamera-lorong.avi"

    done = wait_done(client, snapshot["id"])
    assert done["state"] == "completed", done.get("error")
    assert done["alerts"], "alarm dari event engine harus sampai ke snapshot"
    alert = done["alerts"][0]
    assert alert["track_id"] == "1" and alert["label"] == "FallDown"
    assert isinstance(alert["timestamp_ms"], int)
    assert done["stats"]["tracked_people"] == 1
    assert "is_fall" in done["stats"] and "inference_fps" in done["stats"]

    # Detektor dipanggil dengan kelas dari konfigurasi, bukan hard-coded.
    yolo = FakeYOLO.instances[0]
    assert yolo.track_calls and all(call["classes"] == [0] for call in yolo.track_calls)

    # Riwayat: kolom yang dibaca UI harus ada walau CSV engine memakai nama lain.
    folder = done["output"]
    detail = client.get(f"/api/sessions/{folder}").json()
    assert detail["live"] is False
    assert detail["alerts"][0]["timestamp_ms"] not in ("", None)
    assert detail["predictions"] and detail["predictions"][0]["timestamp_ms"] not in ("", None)
    assert detail["summary"]["detector_source"] == "local"
    assert detail["summary"]["source"] == "kamera-lorong.avi"
    listing = client.get("/api/sessions").json()
    assert listing[0]["id"] == folder and listing[0]["alerts_count"] >= 1


def test_wrong_class_id_fails_with_clear_message(env, monkeypatch):
    monkeypatch.setattr(person_engine, "resolve_detector",
                        lambda **k: ResolvedDetector(env / "best.pt", "local", "best.pt", [7], 0.1))
    video = env / "uji.avi"
    make_video(video, seconds=3)
    client = TestClient(main.app)
    with video.open("rb") as handle:
        snapshot = client.post("/api/sessions/start", data={"save_video": "false", "realtime": "false", "analysis_scope": "person"},
                               files={"file": ("a.avi", handle, "video/x-msvideo")}).json()
    done = wait_done(client, snapshot["id"])
    assert done["state"] == "failed"
    assert "person_class_ids" in done["error"] and "person" in done["error"]
