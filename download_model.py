"""Fetch official Ultralytics YOLO11n-Pose weights for local use."""
from pathlib import Path
import shutil
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yolo11n-pose.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"


def download_model():
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"Model tersedia: {MODEL_PATH}")
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".download")
    try:
        with urlopen(MODEL_URL, timeout=120) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError("Unduhan model tidak lengkap. Jalankan ulang MULAI.cmd.")
        temporary.replace(MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Model disimpan: {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    download_model()
