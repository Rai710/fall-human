"""Pilih model deteksi orang (bounding box): pretrained / best.pt lokal / Hugging Face.

Konfigurasi dibaca dari ``model_config.json`` lalu ditimpa environment variable.
Tidak ada model yang dimuat di sini; modul hanya memastikan file bobot tersedia
dan mengembalikan path-nya. Mengimpor modul ini aman tanpa torch/ultralytics.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "model_config.json"
MODELS_DIR = ROOT / "models"
SOURCES = ("pretrained", "local", "huggingface")

DEFAULTS = {
    "source": "pretrained",
    "pretrained": {"name": "yolo11n-pose.pt", "auto_download": True},
    "local": {"path": "models/best.pt"},
    "huggingface": {"repo_id": "", "filename": "best.pt", "revision": "main",
                    "token_env": "HF_TOKEN"},
    "person_class_ids": [0],
    "min_confidence": 0.1,
}


@dataclass(frozen=True)
class ResolvedDetector:
    path: Path
    source: str
    label: str
    classes: list | None
    min_confidence: float


def _merge(base, extra):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in (extra or {}).items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update({k: v for k, v in value.items() if not k.startswith("_")})
        else:
            out[key] = value
    return out


def parse_classes(value):
    """'0' -> [0]; '0,2' -> [0, 2]; 'all'/'none'/'' -> None; list -> list[int]."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip().lower()
    if text in ("", "all", "none", "null", "*"):
        return None
    return [int(part) for part in re.split(r"[,\s]+", text) if part]


def load_config(path: Path | None = None, env=None) -> dict:
    env = os.environ if env is None else env
    path = Path(path or CONFIG_PATH)
    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name} bukan JSON yang valid: {error}") from error
    cfg = _merge(DEFAULTS, data)
    if env.get("DETECTOR_SOURCE"):
        cfg["source"] = env["DETECTOR_SOURCE"].strip().lower()
    if env.get("DETECTOR_LOCAL_PATH"):
        cfg["local"]["path"] = env["DETECTOR_LOCAL_PATH"]
    if env.get("DETECTOR_HF_REPO"):
        cfg["huggingface"]["repo_id"] = env["DETECTOR_HF_REPO"].strip()
    if env.get("DETECTOR_HF_FILENAME"):
        cfg["huggingface"]["filename"] = env["DETECTOR_HF_FILENAME"].strip()
    if env.get("DETECTOR_HF_REVISION"):
        cfg["huggingface"]["revision"] = env["DETECTOR_HF_REVISION"].strip()
    if "DETECTOR_CLASSES" in env:
        cfg["person_class_ids"] = parse_classes(env["DETECTOR_CLASSES"])
    else:
        cfg["person_class_ids"] = parse_classes(cfg.get("person_class_ids"))
    if cfg["source"] not in SOURCES:
        raise ValueError(f"source harus salah satu dari {SOURCES}, bukan '{cfg['source']}'.")
    cfg["min_confidence"] = float(cfg["min_confidence"])
    if not 0 < cfg["min_confidence"] < 1:
        raise ValueError("min_confidence harus di antara 0 dan 1.")
    return cfg


def _local_path(cfg) -> Path:
    path = Path(cfg["local"]["path"]).expanduser()
    return path if path.is_absolute() else ROOT / path


def _hf_target(cfg) -> Path:
    hf = cfg["huggingface"]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", hf["repo_id"])
    return MODELS_DIR / "hf" / slug


def describe_detector(cfg: dict | None = None) -> dict:
    """Ringkasan untuk UI/API. Tidak mengunduh apa pun."""
    cfg = cfg or load_config()
    source = cfg["source"]
    if source == "pretrained":
        name = cfg["pretrained"]["name"]
        path = MODELS_DIR / name
        label, ready = name, path.is_file()
    elif source == "local":
        path = _local_path(cfg)
        label, ready = path.name, path.is_file()
    else:
        hf = cfg["huggingface"]
        path = _hf_target(cfg) / hf["filename"] if hf["repo_id"] else None
        label = f"{hf['repo_id'] or '(repo_id kosong)'}/{hf['filename']}"
        ready = bool(path and path.is_file())
    return {"source": source, "label": label, "ready": ready,
            "classes": cfg["person_class_ids"], "min_confidence": cfg["min_confidence"]}


def _download_pretrained(name: str, target: Path, on_status):
    on_status(f"Mengunduh {name} dari rilis Ultralytics (sekali saja)...")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(target))
    except Exception as error:  # jaringan, nama tidak dikenal, dsb.
        raise RuntimeError(
            f"Gagal mengunduh {name}: {error}. Unduh manual dari "
            f"https://github.com/ultralytics/assets/releases lalu taruh di {target}") from error
    if not target.is_file():
        raise RuntimeError(f"Unduhan {name} tidak menghasilkan file di {target}.")


def resolve_detector(cfg: dict | None = None, on_status=None) -> ResolvedDetector:
    """Kembalikan path bobot YOLO sesuai konfigurasi; unduh bila perlu."""
    on_status = on_status or (lambda text: None)
    cfg = cfg or load_config()
    source = cfg["source"]
    classes, min_conf = cfg["person_class_ids"], cfg["min_confidence"]

    if source == "pretrained":
        name = cfg["pretrained"]["name"]
        path = MODELS_DIR / name
        if not path.is_file():
            if not cfg["pretrained"].get("auto_download", True):
                raise FileNotFoundError(
                    f"{path} tidak ada dan auto_download=false. "
                    "Jalankan `python download_detector.py` atau taruh file-nya di folder models/.")
            _download_pretrained(name, path, on_status)
        return ResolvedDetector(path, source, name, classes, min_conf)

    if source == "local":
        path = _local_path(cfg)
        if not path.is_file():
            raise FileNotFoundError(
                f"best.pt lokal tidak ditemukan: {path}. Salin model hasil train ke sana "
                "atau ubah local.path di model_config.json.")
        return ResolvedDetector(path, source, path.name, classes, min_conf)

    hf = cfg["huggingface"]
    if not hf["repo_id"]:
        raise ValueError("huggingface.repo_id masih kosong di model_config.json "
                         "(contoh: \"username/nama-repo-model\").")
    target = _hf_target(cfg)
    path = target / hf["filename"]
    token = os.environ.get(hf.get("token_env") or "HF_TOKEN") or None
    from huggingface_hub import hf_hub_download
    on_status(f"Mengambil {hf['filename']} dari Hugging Face ({hf['repo_id']})...")
    try:
        downloaded = hf_hub_download(repo_id=hf["repo_id"], filename=hf["filename"],
                                     revision=hf["revision"], token=token,
                                     local_dir=str(target))
    except Exception as error:
        if path.is_file():  # offline tapi salinan lama masih ada
            on_status("Hugging Face tidak terjangkau; memakai salinan terunduh sebelumnya.")
            downloaded = str(path)
        else:
            raise RuntimeError(
                f"Gagal mengambil model dari Hugging Face: {error}. Untuk repo privat "
                f"set environment variable {hf.get('token_env') or 'HF_TOKEN'}.") from error
    return ResolvedDetector(Path(downloaded), source, f"{hf['repo_id']}/{hf['filename']}",
                            classes, min_conf)
