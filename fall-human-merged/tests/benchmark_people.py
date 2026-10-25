"""Local sample benchmark. Not a hospital accuracy evaluation or webcam test."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from person_engine import process_people


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", help="Only these sample filenames (repeatable).")
    parser.add_argument("--mode", choices=("offline", "replay", "both"), default="both")
    args = parser.parse_args()
    project = args.project_root
    output = args.output or project / "outputs" / "benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    report = []
    cases = [("blank_test.mp4", False), ("gmdcsa_fall_01.mp4", False),
             ("gmdcsa_adl_02.mp4", False), ("multi_person_test.mp4", False),
             ("gmdcsa_fall_01.mp4", True), ("multi_person_test.mp4", True)]
    cases = [(name, realtime) for name, realtime in cases
             if (not args.case or name in args.case)
             and (args.mode == "both" or realtime == (args.mode == "replay"))]
    for name, realtime in cases:
        print(f"START {name} {'replay' if realtime else 'offline'}", flush=True)
        stamps = []
        def preview(frame, stats):
            stamps.append(time.monotonic())
        folder, summary = process_people(
            project / "data" / name, realtime=realtime, save_video=True,
            output_root=output, model_path=project / "models/videomae-fall",
            detector_path=project / "models/yolo11n-pose.pt", local_only=True,
            on_frame=preview,
            on_event=lambda kind, payload: print(kind, payload, flush=True))
        rows = list(csv.DictReader((folder / "predictions.csv").open(encoding="utf-8", newline="")))
        measured = [row for row in rows if row["skipped"] == "0"]
        gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
        record = dict(sample=name, mode=summary["mode"], output=str(folder),
            device=summary["device"], frames=summary["frames_processed"],
            peak_people=summary.get("peak_people"), unique_ids=summary.get("unique_track_ids"),
            detector_mean_ms=summary.get("detector_mean_ms"), detector_p95_ms=summary.get("detector_p95_ms"),
            analysis_mean_ms=summary["analysis_mean_ms"], analysis_p95_ms=summary["analysis_p95_ms"],
            result_age_p95_ms=summary["result_age_p95_ms"], queue_p95_ms=summary["queue_p95_ms"],
            analyses=summary["analyses"], stale=summary["stale_results"],
            replaced=summary["replaced_clips"], expired=summary["expired_clips"],
            tracking_drops=summary["replaced_tracking_frames"], resets=summary["clip_segment_resets"],
            alerts=summary["alerts_count"], status=summary["status"],
            maximum_preview_gap_ms=max(gaps, default=0),
            labels=sorted({row["label"] for row in measured}),
            max_fall_score=max((float(row["fall_score"]) for row in measured), default=None),
            warnings=summary["warnings"])
        assert record["status"] == "completed"
        assert summary["recorded_frames"] == summary["frames_processed"]
        assert len(measured) == summary["analyses"]
        if name == "blank_test.mp4":
            assert record["alerts"] == 0 and record["analyses"] == 0
        if name == "multi_person_test.mp4":
            assert record["peak_people"] >= 2, "Multi-person fixture should detect multiple people"
        report.append(record)
        (output / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("RESULT", json.dumps(record), flush=True)
    print("REPORT", output / "benchmark_report.json", flush=True)


if __name__ == "__main__":
    main()
