"""Unduh model yang dibutuhkan sebelum aplikasi dijalankan. Aman dipanggil berulang.

File ini juga menyimpan nama model dan lokasi file model supaya pose_engine.py
dan skrip ini memakai nilai yang sama. Hanya memakai pustaka standar di tingkat
atas, jadi bisa diimpor tanpa memuat torch.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent

# Model penentu jatuh (Hugging Face). Lisensi model: CC BY-NC 4.0 (non-komersial).
MODEL_NAME = "yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect"

# Model kerangka MediaPipe (hanya untuk menggambar, tidak ikut menentukan jatuh).
POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker_lite.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")


def ensure_pose_model() -> Path:
    """Unduh model kerangka bila belum ada. Melempar galat jika unduhan gagal."""
    if POSE_MODEL_PATH.is_file() and POSE_MODEL_PATH.stat().st_size > 1_000_000:
        return POSE_MODEL_PATH
    POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = POSE_MODEL_PATH.with_suffix(".download")
    try:
        with urlopen(POSE_MODEL_URL, timeout=120) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Unduhan model kerangka tidak lengkap.")
        temporary.replace(POSE_MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return POSE_MODEL_PATH


def fetch_video_model() -> None:
    """Unduh (atau cek cache) model VideoMAE supaya pemuatan pertama di aplikasi tidak lama."""
    from transformers import AutoImageProcessor, AutoModelForVideoClassification
    AutoImageProcessor.from_pretrained(MODEL_NAME)
    AutoModelForVideoClassification.from_pretrained(MODEL_NAME)


def main() -> int:
    code = 0
    print("Menyiapkan model penentu jatuh (VideoMAE)...")
    try:
        fetch_video_model()
        print("Model VideoMAE siap.")
    except Exception as error:
        print(f"GAGAL menyiapkan model VideoMAE: {error}")
        code = 1
    print("Menyiapkan model kerangka (MediaPipe)...")
    try:
        print(f"Model kerangka siap: {ensure_pose_model()}")
    except Exception as error:
        # Kerangka hanya hiasan; deteksi jatuh tetap jalan tanpanya.
        print(f"PERINGATAN: model kerangka tidak bisa diunduh ({error}). "
              "Aplikasi tetap jalan tanpa kerangka.")
    return code


if __name__ == "__main__":
    sys.exit(main())
