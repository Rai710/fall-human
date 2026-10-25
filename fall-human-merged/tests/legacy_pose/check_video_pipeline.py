"""Integration checks: multiple people, blank frames, and interrupted exports.

The scene fixture is generated from an official Ultralytics example image. It
checks pipeline wiring and tracking continuity, not hospital accuracy.
"""
import csv
from itertools import groupby
import json
from pathlib import Path
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_engine import ROOT, process_source


def check_export(session, summary):
    frames = list(csv.DictReader((session / "frames.csv").open(newline="", encoding="utf-8")))
    detections = list(csv.DictReader((session / "detections.csv").open(newline="", encoding="utf-8")))
    detection_counts = {}
    for detection in detections:
        frame_index = int(detection["frame_index"])
        detection_counts[frame_index] = detection_counts.get(frame_index, 0) + 1
        assert int(detection["timestamp_ms"]) == int(frames[frame_index]["timestamp_ms"])
        assert 0 <= float(detection["bbox_x1"]) <= float(detection["bbox_x2"]) <= 1
        assert 0 <= float(detection["bbox_y1"]) <= float(detection["bbox_y2"]) <= 1
    alerts = list(csv.DictReader((session / "alerts.csv").open(newline="", encoding="utf-8")))
    assert len(alerts) == summary["alerts_count"]
    assert all(alert["event"] == "suspected_collapse" and int(alert["track_id"]) > 0 for alert in alerts)
    assert all(int(alert["horizontal_hold_ms"]) >= 1500 for alert in alerts)
    assert len(frames) == summary["frames_processed"] > 0
    assert all(detection_counts.get(index, 0) == int(frame["detections"])
               for index, frame in enumerate(frames))
    times = [int(frame["timestamp_ms"]) for frame in frames]
    assert all(after > before for before, after in zip(times, times[1:]))
    by_track = {}
    with (session / "landmarks.csv").open(newline="", encoding="utf-8") as source:
        groups = groupby(csv.DictReader(source), key=lambda row: (row["frame_index"], row["track_id"]))
        for (frame_index, identity), items in groups:
            rows = list(items)
            assert len(rows) == 17
            assert {int(row["landmark_index"]) for row in rows} == set(range(17))
            assert len({row["timestamp_ms"] for row in rows}) == 1
            by_track.setdefault(int(identity), []).extend(rows)
            for row in rows:
                assert int(row["timestamp_ms"]) == times[int(frame_index)]
                if row["observed"] == "0":
                    assert row["x"] == row["confidence"] == ""
    for identity, expected in by_track.items():
        with (session / "tracks" / f"track_{identity:04d}.csv").open(newline="", encoding="utf-8") as source:
            actual = list(csv.DictReader(source))
        assert expected == actual
        assert all(int(row["track_id"]) == identity for row in actual)
    assert set(map(str, by_track)) == set(map(str, summary["tracks"]))
    assert json.loads((session / "summary.json").read_text()) == json.loads(json.dumps(summary))
    if summary["annotated_video"]:
        capture = cv2.VideoCapture(str(session / summary["annotated_video"]))
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        assert count == len(frames)
    return frames, by_track


def make_scene(image_path):
    image = cv2.imread(str(image_path))
    assert image is not None
    height, width = image.shape[:2]
    scene = ROOT / "data" / "multi_person_test.mp4"
    writer = cv2.VideoWriter(str(scene), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    assert writer.isOpened()
    for index in range(24):
        shift = np.float32([[1, 0, index % 8], [0, 1, 0]])
        writer.write(cv2.warpAffine(image, shift, (width, height)))
    for _ in range(10):
        writer.write(np.zeros_like(image))
    for _ in range(12):
        writer.write(image)
    writer.release()
    return scene


if __name__ == "__main__":
    scene = make_scene(ROOT / "data" / "ultralytics_bus.jpg")
    session, summary = process_source(scene)
    frames, tracks = check_export(session, summary)
    assert summary["max_simultaneous_tracked"] >= 2, summary
    stable_tracks = [identity for identity, rows in tracks.items()
                     if len({row["frame_index"] for row in rows if row["observed"] == "1"
                             and int(row["frame_index"]) < 24}) >= 20]
    assert len(stable_tracks) >= 2, stable_tracks
    assert all(int(frame["tracked_people"]) == 0 for frame in frames[24:34])
    report = [{"source": scene.name, "output": str(session),
               "frames": summary["frames_processed"], "max_people": summary["max_simultaneous_tracked"],
               "stable_ids_first_24_frames": stable_tracks, "inference_fps": summary["inference_fps"]}]
    for video in sys.argv[1:]:
        session, summary = process_source(Path(video))
        check_export(session, summary)
        assert summary["frames_with_pose"] > 0
        if Path(video).name == "gmdcsa_fall_01.mp4":
            assert summary["alerts_count"] >= 1, "Klip contoh jatuh seharusnya memberi alarm."
        if Path(video).name == "gmdcsa_adl_02.mp4":
            assert summary["alerts_count"] == 0, "Klip berbaring ke ranjang tidak seharusnya memberi alarm."
        report.append({"source": Path(video).name, "output": str(session),
                       "frames": summary["frames_processed"], "pose_frames": summary["frames_with_pose"],
                       "max_people": summary["max_simultaneous_tracked"],
                       "alerts": summary["alerts_count"], "inference_fps": summary["inference_fps"]})
    blank = ROOT / "data" / "yolo_blank_test.mp4"
    writer = cv2.VideoWriter(str(blank), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    session, summary = process_source(blank, save_video=False)
    check_export(session, summary)
    assert summary["unique_track_ids"] == 0
    stop = threading.Event()
    def stop_after_three(frame, stats):
        if stats["frame"] == 3:
            stop.set()
    session, summary = process_source(blank, save_video=False, stop=stop, on_frame=stop_after_three)
    check_export(session, summary)
    assert summary["status"] == "stopped" and summary["frames_processed"] == 3
    report.append({"blank_and_stop_checks": "passed"})
    (ROOT / "outputs" / "yolo_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
