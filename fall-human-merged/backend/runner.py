from __future__ import annotations

import os
import shutil
import subprocess
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


def _ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - opsional
        return None


def make_browser_playable(video: Path) -> bool:
    """OpenCV menulis mp4v, yang tidak diputar <video> di Chrome/Edge. Ubah ke H.264."""
    exe = _ffmpeg_exe()
    if exe is None or not video.is_file():
        return False
    temporary = video.with_name("annotated.h264.tmp.mp4")
    command = [exe, "-y", "-loglevel", "error", "-i", str(video), "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
               "-movflags", "+faststart", str(temporary)]
    try:
        subprocess.run(command, check=True, timeout=900)
        temporary.replace(video)
        return True
    except (OSError, subprocess.SubprocessError):
        temporary.unlink(missing_ok=True)
        return False


class SessionRunner:
    def __init__(self, source: Path | int, save_video: bool = True,
                 realtime: bool = True, name: str | None = None):
        self.id = uuid.uuid4().hex
        self.source = source
        self.save_video = save_video
        self.realtime = realtime
        self.name = (name or (f"Webcam {source}" if isinstance(source, int) else "Video")).strip()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"session-{self.id[:8]}", daemon=True)
        self.lock = threading.Lock()
        self.state = "starting"
        self.error: str | None = None
        self.message = "Memulai sesi..."
        self.device = "cpu"
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
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001
                pass
            from person_engine import process_people

            out_dir, summary = process_people(
                self.source,
                save_video=self.save_video,
                stop=self.stop_event,
                on_frame=self._on_frame,
                on_event=self._on_event,
                realtime=self.realtime,
                camera_name=self.name,
                # Model VideoMAE (HF) diunduh otomatis pada sesi pertama bila belum ada.
                local_only=False,
            )
            if self.save_video and not isinstance(self.source, int):
                video = Path(out_dir) / "annotated.mp4"
                if video.is_file():
                    with self.lock:
                        self.message = "Mengonversi video anotasi agar bisa diputar di browser..."
                    if not make_browser_playable(video):
                        summary.setdefault("warnings", []).append(
                            "Video anotasi memakai codec mp4v dan mungkin tidak diputar browser. "
                            "Pasang ffmpeg (atau `pip install imageio-ffmpeg`) lalu mulai ulang.")
            with self.lock:
                self.output_dir = Path(out_dir)
                self.summary = summary
                self.state = summary.get("status", "completed")
                self.message = ""
                self.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.state = "failed"
                self.error = str(exc)
                self.finished_at = time.time()

    def _on_event(self, kind: str, payload) -> None:
        with self.lock:
            if kind == "session":
                self.output_dir = Path(payload)
            elif kind == "status":
                self.message = str(payload)
            elif kind == "alarm":
                self.alerts.append({
                    "alert_id": str(payload.get("alert_id", len(self.alerts) + 1)),
                    "track_id": str(payload.get("track_id", "")),
                    "segment_id": str(payload.get("segment_id", "")),
                    "timestamp_ms": payload.get("delivered_source_ms", payload.get("clip_end_ms")),
                    "label": payload.get("label"),
                    "confidence": payload.get("confidence"),
                    "fall_score": payload.get("fall_score"),
                })

    def _on_frame(self, frame, stats: dict) -> None:
        now = time.monotonic()
        jpeg = None
        if now - self.last_preview_at >= PREVIEW_INTERVAL:
            jpeg = _encode_jpeg(frame)
            self.last_preview_at = now
        mapped = dict(stats)
        mapped.update(
            inference_fps=stats.get("capture_fps"),
            is_fall=stats.get("active_alarms", 0) > 0,
            infer_busy=stats.get("pending_people", 0) > 0,
            last_result_age_ms=stats.get("result_age_ms"),
            device=self.device,
        )
        with self.lock:
            if jpeg is not None:
                self.latest_frame = jpeg
            self.stats = mapped
            self.state = "running"
            self.message = ""

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
                "message": self.message,
                "source": str(self.source),
                "name": self.name,
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

    def start(self, source: Path | int, save_video: bool = True,
              realtime: bool = True, name: str | None = None) -> SessionRunner:
        with self.lock:
            if self.active_id:
                current = self.sessions.get(self.active_id)
                if current and current.state in ("starting", "running"):
                    raise RuntimeError("Masih ada sesi yang berjalan. Hentikan sesi tersebut terlebih dahulu.")
            runner = SessionRunner(source, save_video=save_video, realtime=realtime, name=name)
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
