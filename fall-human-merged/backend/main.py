from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Literal

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import FRONTEND_DIST, UPLOADS_DIR
from . import repository
from detector_config import describe_detector, load_config
from action_config import describe_action
from .runner import SessionManager

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

app = FastAPI(title="Fall Detection Dashboard", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
manager = SessionManager()


class ReviewPayload(BaseModel):
    acked: list[str] = Field(default_factory=list)
    labels: dict[str, str] | None = None


def _folder_for(session_id: str) -> str:
    runner = manager.get(session_id)
    if runner is not None:
        if runner.output_dir is None:
            raise HTTPException(409, "Folder sesi belum siap.")
        return runner.output_dir.name
    return session_id


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    """Model deteksi orang yang aktif (dari model_config.json). Tidak mengunduh apa pun."""
    try:
        return {"detector": describe_detector(load_config()), "action_model": describe_action()}
    except ValueError as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/sessions/start")
async def start_session(
    file: UploadFile | None = File(None),
    camera_index: str = Form(""),
    save_video: bool = Form(True),
    realtime: bool = Form(False),
    analysis_scope: Literal["scene", "person"] = Form("scene"),
) -> dict:
    upload_path: Path | None = None
    source: Path | int | None = None
    display_name: str | None = None

    if file is not None and file.filename:
        extension = Path(file.filename).suffix.lower()
        if extension not in VIDEO_EXTENSIONS:
            raise HTTPException(400, f"Format video tidak didukung: {extension or '(tanpa ekstensi)'}")
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        upload_path = UPLOADS_DIR / f"{uuid.uuid4().hex}{extension}"
        size = 0
        try:
            with upload_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Ukuran video melebihi 500 MB.")
                    output.write(chunk)
        except Exception:
            upload_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        source = upload_path
        display_name = Path(file.filename).name[:80]
    elif camera_index.strip():
        try:
            index = int(camera_index)
        except ValueError as exc:
            raise HTTPException(400, "Nomor webcam harus berupa angka.") from exc
        if index < 0:
            raise HTTPException(400, "Nomor webcam tidak valid.")
        source = index
    else:
        raise HTTPException(400, "Pilih file video atau isi nomor webcam.")

    try:
        runner = manager.start(source, save_video=save_video, realtime=realtime, name=display_name, analysis_scope=analysis_scope)
    except RuntimeError as exc:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)
        raise HTTPException(409, str(exc)) from exc
    return runner.snapshot()


@app.post("/api/sessions/{session_id}/stop")
def stop_session(session_id: str) -> dict:
    snapshot = manager.stop(session_id)
    if snapshot is None:
        raise HTTPException(404, "Sesi live tidak ditemukan.")
    return snapshot


@app.get("/api/sessions/{session_id}/stream")
async def stream_session(session_id: str, request: Request) -> StreamingResponse:
    runner = manager.get(session_id)
    if runner is None:
        raise HTTPException(404, "Sesi live tidak ditemukan.")

    async def frames():
        while True:
            if await request.is_disconnected():
                break
            jpeg = runner.jpeg()
            if jpeg:
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                yield header + jpeg + b"\r\n"
            if runner.state not in ("starting", "running", "stopping"):
                break
            await asyncio.sleep(1 / 15)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/api/sessions/{session_id}/events")
async def session_events(websocket: WebSocket, session_id: str) -> None:
    runner = manager.get(session_id)
    if runner is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    try:
        while True:
            await websocket.send_json(runner.snapshot())
            if runner.state not in ("starting", "running", "stopping"):
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    return repository.list_sessions()


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str) -> dict:
    runner = manager.get(session_id)
    if runner is not None:
        return {"live": True, **runner.snapshot()}
    try:
        return {"live": False, **repository.get_session(session_id)}
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/sessions/{session_id}/review")
def save_review(session_id: str, payload: ReviewPayload) -> dict:
    folder = _folder_for(session_id)
    try:
        return repository.save_review(folder, payload.acked, payload.labels)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404 if isinstance(exc, FileNotFoundError) else 400, str(exc)) from exc


@app.get("/api/sessions/{session_id}/download")
def download_session(session_id: str) -> Response:
    folder = _folder_for(session_id)
    try:
        content = repository.zip_session(folder)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(
        content=content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{folder}.zip"'},
    )


@app.get("/api/sessions/{session_id}/files/{filename}")
def session_file(session_id: str, filename: str) -> FileResponse:
    folder = _folder_for(session_id)
    try:
        path = repository.safe_file(folder, filename)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path)


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @app.get("/")
    def root() -> dict:
        return {"message": "Backend aktif. Jalankan `npm run dev` di folder frontend atau build frontend."}
