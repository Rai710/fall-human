"""Run the full pose/tracker/rule path on one fall and one non-fall clip."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_engine import DEFAULT_MODEL, ROOT, process_source
from replay_existing_csv import replay


def run(video, model, output):
    callback_alerts = []
    def on_frame(_frame, stats):
        callback_alerts.extend(stats["new_alerts"])
    session, summary = process_source(video, output_root=output, model_path=model,
                                      save_video=True, on_frame=on_frame)
    with (session / "alerts.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == summary["alerts_count"] == len(callback_alerts)
    assert all(int(row["horizontal_hold_ms"]) >= 1500 for row in rows)
    replayed = replay(session)
    assert [event["frame_index"] for event in replayed] == [int(row["frame_index"]) for row in rows]
    return {"session": str(session), "source": str(video), "alerts": len(rows),
            "event_frames": [int(row["frame_index"]) for row in rows],
            "frames": summary["frames_processed"], "inference_fps": summary["inference_fps"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fall", type=Path)
    parser.add_argument("non_fall", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    fall = run(args.fall, args.model, args.output)
    non_fall = run(args.non_fall, args.model, args.output)
    assert fall["alerts"] == 1, fall
    assert non_fall["alerts"] == 0, non_fall
    print(json.dumps([fall, non_fall], indent=2))
