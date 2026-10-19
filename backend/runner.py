from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

from . import PROJECT_ROOT

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import cv2

PREVIEW_INTERVAL = 1 / 15.0


def _encode_jpeg(frame) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return buffer.tobytes() if ok else None


class SessionRunner:
    def __init__(self, source: Path | int, save_video: bool = True):
        self.id = uuid.uuid4().hex
        self.source = source
        self.save_video = save_video
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"session-{self.id[:8]}", daemon=True)
        self.lock = threading.Lock()
        self.state = "starting"
        self.error: str | None = None
        self.stats: dict = {}
        self.alerts: list[dict] = []
        self.output_dir: Path | None = None
        self.summary: dict | None = None
        self.latest_frame: bytes | None = None
        self.last_preview_at = 0.0
        self.started_at = time.time()
        self.finished_at: float | None = None

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            from pose_engine import process_source

            out_dir, summary = process_source(
                self.source,
                save_video=self.save_video,
                stop=self.stop_event,
                on_frame=self._on_frame,
                on_session=self._on_session,
            )
            with self.lock:
                self.output_dir = Path(out_dir)
                self.summary = summary
                self.state = summary.get("status", "completed")
                self.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.state = "failed"
                self.error = str(exc)
                self.finished_at = time.time()

    def _on_session(self, out_dir: Path) -> None:
        with self.lock:
            self.output_dir = Path(out_dir)

    def _on_frame(self, frame, stats: dict) -> None:
        now = time.monotonic()
        jpeg = None
        if now - self.last_preview_at >= PREVIEW_INTERVAL:
            jpeg = _encode_jpeg(frame)
            self.last_preview_at = now
        new_alerts = stats.get("new_alerts") or []
        with self.lock:
            if jpeg is not None:
                self.latest_frame = jpeg
            self.stats = dict(stats)
            self.state = "running"
            for alert in new_alerts:
                self.alerts.append({
                    "alert_id": str(alert.get("track_id", len(self.alerts) + 1)),
                    "timestamp_ms": alert.get("timestamp_ms"),
                    "label": alert.get("label"),
                    "confidence": alert.get("confidence"),
                })

    def stop(self, timeout: float = 30.0) -> dict:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout)
        return self.snapshot()

    def jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_frame

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "state": self.state,
                "error": self.error,
                "source": str(self.source),
                "output": self.output_dir.name if self.output_dir else None,
                "stats": dict(self.stats),
                "alerts": list(self.alerts),
                "summary": self.summary,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


class SessionManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sessions: dict[str, SessionRunner] = {}
        self.active_id: str | None = None

    def start(self, source: Path | int, save_video: bool = True) -> SessionRunner:
        with self.lock:
            if self.active_id:
                current = self.sessions.get(self.active_id)
                if current and current.state in ("starting", "running"):
                    raise RuntimeError("Masih ada sesi yang berjalan. Hentikan sesi tersebut terlebih dahulu.")
            runner = SessionRunner(source, save_video=save_video)
            self.sessions[runner.id] = runner
            self.active_id = runner.id
        runner.start()
        return runner

    def get(self, session_id: str) -> SessionRunner | None:
        with self.lock:
            return self.sessions.get(session_id)

    def stop(self, session_id: str) -> dict | None:
        runner = self.get(session_id)
        if runner is None:
            return None
        snapshot = runner.stop()
        with self.lock:
            if self.active_id == session_id:
                self.active_id = None
        return snapshot
