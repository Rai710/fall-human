"""Metadata and RGB preprocessing for the exported Le2i + CAUCAFall checkpoint."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

MIXED_LABEL = "Le2i + CAUCAFall (2 kelas)"
MIXED_FOLDER = "best_model-20261025T045257Z-1-001"


def mixed_model_path(root):
    return Path(root)/"models"/MIXED_FOLDER/"best_model"


def read_mixed_settings(model_dir):
    folder=Path(model_dir)
    settings=json.loads((folder/"inference_config.json").read_text(encoding="utf-8"))
    config=json.loads((folder/"config.json").read_text(encoding="utf-8"))
    marker=json.loads((folder/"export_complete.json").read_text(encoding="utf-8"))
    if marker.get("ready") is not True:
        raise ValueError("Ekspor Le2i + CAUCAFall belum lengkap.")
    expected={"0":"no_fall","1":"fall"}
    if config.get("model_type")!="videomae" or config.get("id2label")!=expected or settings.get("id2label")!=expected:
        raise ValueError("Checkpoint Le2i + CAUCAFall harus memiliki kelas no_fall dan fall.")
    if any(obj.get("num_frames")!=16 or obj.get("image_size")!=224 for obj in (settings,config)):
        raise ValueError("Checkpoint memerlukan 16 frame RGB 224x224.")
    if settings.get("clip_seconds")!=2. or settings.get("resize")!="PIL_letterbox_black_bilinear":
        raise ValueError("Preprocessing ekspor berbeda dari profil model dua kelas.")
    if not 0<float(settings["threshold"])<=1 or float(settings["stride_seconds"])<.5:
        raise ValueError("Threshold atau stride ekspor tidak valid.")
    return settings


def verify_mixed_export(model_dir):
    folder=Path(model_dir)
    marker=json.loads((folder/"export_complete.json").read_text(encoding="utf-8"))
    for name,expected in marker["files"].items():
        path=folder/name
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Path file ekspor tidak valid.")
        digest=hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(4*1024*1024),b""):
                digest.update(block)
        if digest.hexdigest()!=expected:
            raise ValueError("File ekspor berubah atau tidak lengkap: "+name)
    return marker


def mixed_rgb_image(frame):
    """Input is RGB; capture converts BGR before calling this function."""
    image=frame if isinstance(frame,Image.Image) else Image.fromarray(frame)
    image=image.convert("RGB")
    return ImageOps.pad(image,(224,224),method=Image.Resampling.BILINEAR,
                        color=(0,0,0),centering=(.5,.5))
