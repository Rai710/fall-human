"""Replay exported pose histories through the collapse rule without rerunning YOLO.

Usage: python tests/replay_existing_csv.py <session-folder> [<session-folder> ...]
The existing CSV keeps the original source timestamps and quality flags.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fall_rules import CollapseDetector, observation
from pose_engine import Person, TrackFrame


def replay(session, diagnose=False):
    by_frame = {}
    with (session / "landmarks.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            by_frame.setdefault(int(row["frame_index"]), {}).setdefault(int(row["track_id"]), []).append(row)
    unassigned_by_frame = {}
    detections_path = session / "detections.csv"
    if detections_path.exists():
        with detections_path.open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row["track_id"] == "":
                    unassigned_by_frame.setdefault(int(row["frame_index"]), []).append(row)
    detector = CollapseDetector()
    alerts = []
    previous_status = {}
    with (session / "frames.csv").open(newline="", encoding="utf-8") as source:
        for frame in csv.DictReader(source):
            index = int(frame["frame_index"])
            width, height = int(frame["width"]), int(frame["height"])
            records = []
            for identity, rows in by_frame.get(index, {}).items():
                first = rows[0]
                person = None
                if first["observed"] == "1":
                    points = np.zeros((17, 3), dtype=float)
                    for row in rows:
                        point_index = int(row["landmark_index"])
                        points[point_index] = (
                            float(row["x"]) * width if row["x"] else np.nan,
                            float(row["y"]) * height if row["y"] else np.nan,
                            float(row["confidence"]) if row["confidence"] else 0,
                        )
                    bbox = (float(first["bbox_x1"]) * width, float(first["bbox_y1"]) * height,
                            float(first["bbox_x2"]) * width, float(first["bbox_y2"]) * height)
                    person = Person(identity, bbox, float(first["detection_confidence"]), points)
                records.append(TrackFrame(identity, int(first["segment_id"]), person,
                                          first["pose_usable"] == "1"))
            unassigned = []
            for row in unassigned_by_frame.get(index, []):
                bbox = (float(row["bbox_x1"]) * width, float(row["bbox_y1"]) * height,
                        float(row["bbox_x2"]) * width, float(row["bbox_y2"]) * height)
                unassigned.append(Person(None, bbox, float(row["detection_confidence"]),
                                         np.zeros((17, 3), dtype=float)))
            statuses, new_alerts = detector.update(records, int(frame["timestamp_ms"]), width, height,
                                                    index, unassigned_people=unassigned)
            alerts.extend(new_alerts)
            if diagnose:
                for record in records:
                    status = statuses[record.track_id]
                    if status != previous_status.get(record.track_id) or index % 10 == 0:
                        item = observation(record, int(frame["timestamp_ms"]), width, height)
                        if item is not None:
                            print(f"f={index:3} t={item.timestamp_ms/1000:4.2f} id={record.track_id} "
                                  f"status={status:9} pose={int(record.pose_usable)} "
                                  f"shoulder={item.shoulder_y/height if item.shoulder_y is not None else None} "
                                  f"nose={item.nose_y/height if item.nose_y is not None else None} "
                                  f"angle={item.torso_angle_deg} aspect={item.bbox_aspect:.2f} "
                                  f"bottom={item.bbox_bottom/height:.2f} "
                                  f"upright={item.upright} horizontal={item.horizontal}")
                    previous_status[record.track_id] = status
    return alerts


if __name__ == "__main__":
    verbose = "--diagnose" in sys.argv
    for argument in [value for value in sys.argv[1:] if value != "--diagnose"]:
        folder = Path(argument)
        print(json.dumps({"session": str(folder), "alerts": replay(folder, verbose)}, indent=2))
