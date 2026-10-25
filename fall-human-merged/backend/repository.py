from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

from . import OUTPUTS_DIR

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_FILES = {"annotated.mp4", "predictions.csv", "alerts.csv", "summary.json", "review.json"}
ALLOWED_LABELS = {"", "true_fall", "false_alarm"}


def _session_dir(session_id: str) -> Path:
    if not SAFE_ID.match(session_id):
        raise ValueError("ID sesi tidak valid.")
    root = OUTPUTS_DIR.resolve()
    path = (OUTPUTS_DIR / session_id).resolve()
    if path.parent != root or not path.is_dir():
        raise FileNotFoundError("Sesi tidak ditemukan.")
    return path


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _normalize_row(row: dict) -> dict:
    """Engine per-orang menulis clip_end_ms/delivered_source_ms; UI membaca timestamp_ms."""
    if not row.get("timestamp_ms"):
        row["timestamp_ms"] = row.get("delivered_source_ms") or row.get("clip_end_ms") or ""
    return row


def _normalize_summary(summary: dict) -> dict:
    warnings = summary.get("warnings")
    if warnings and not summary.get("warning"):
        summary["warning"] = " ".join(str(item) for item in warnings)
    name, source = summary.get("camera_name"), str(summary.get("source") or "")
    if name and source and not source.isdigit() and source != "network_camera":
        summary["source"] = name  # nama file asli, bukan uuid hasil upload
    return summary


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [_normalize_row(row) for row in csv.DictReader(handle)]


def list_sessions() -> list[dict]:
    if not OUTPUTS_DIR.is_dir():
        return []
    sessions = []
    for path in OUTPUTS_DIR.iterdir():
        if not path.is_dir() or not (path / "summary.json").is_file():
            continue
        summary = _normalize_summary(_read_json(path / "summary.json", {}))
        alerts = _read_csv(path / "alerts.csv")
        review = _read_json(path / "review.json", {"acked": [], "labels": {}})
        sessions.append({
            "id": path.name,
            "modified_at": path.stat().st_mtime,
            "source": summary.get("source"),
            "model": summary.get("model"),
            "device": summary.get("device"),
            "status": summary.get("status"),
            "frames_processed": summary.get("frames_processed", 0),
            "alerts_count": summary.get("alerts_count", len(alerts)),
            "acknowledged": len(review.get("acked", [])),
            "warning": summary.get("warning"),
            "has_video": (path / "annotated.mp4").is_file(),
        })
    return sorted(sessions, key=lambda item: item["modified_at"], reverse=True)


def get_session(session_id: str) -> dict:
    path = _session_dir(session_id)
    summary = _normalize_summary(_read_json(path / "summary.json", {}))
    review = _read_json(path / "review.json", {"acked": [], "labels": {}})
    return {
        "id": path.name,
        "summary": summary,
        "alerts": _read_csv(path / "alerts.csv"),
        "predictions": _read_csv(path / "predictions.csv"),
        "review": {
            "acked": review.get("acked", []),
            "labels": review.get("labels", {}),
        },
        "files": [name for name in ALLOWED_FILES if (path / name).is_file()],
    }


def save_review(session_id: str, acked: list[str], labels: dict[str, str] | None = None) -> dict:
    path = _session_dir(session_id)
    clean_acked = sorted({str(item) for item in acked})

    if labels is None:
        existing = _read_json(path / "review.json", {})
        labels = existing.get("labels", {})

    clean_labels = {}
    for key, value in labels.items():
        if value not in ALLOWED_LABELS:
            raise ValueError(f"Label tidak valid: {value}")
        clean_labels[str(key)] = value

    review = {"acked": clean_acked, "labels": clean_labels}
    target = path / "review.json"
    temporary = path / "review.json.tmp"
    temporary.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return review


def safe_file(session_id: str, filename: str) -> Path:
    if filename not in ALLOWED_FILES:
        raise FileNotFoundError("File tidak tersedia.")
    path = _session_dir(session_id) / filename
    if not path.is_file():
        raise FileNotFoundError("File tidak tersedia.")
    return path


def zip_session(session_id: str) -> bytes:
    path = _session_dir(session_id)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(ALLOWED_FILES):
            item = path / name
            if item.is_file():
                archive.write(item, arcname=f"{path.name}/{name}")
    return buffer.getvalue()
