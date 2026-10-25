import importlib
import importlib.metadata as md
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parent

# (nama paket pip, nama modul import, wajib?)
PACKAGES = [
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
    ("opencv-python", "cv2", True),
    ("Pillow", "PIL", True),
    ("numpy", "numpy", True),
    ("transformers", "transformers", True),
    ("huggingface-hub", "huggingface_hub", True),
    ("safetensors", "safetensors", True),
    ("ultralytics", "ultralytics", True),
    ("lap", "lap", True),  # dibutuhkan ByteTrack; autoinstall dimatikan di engine
    ("PyYAML", "yaml", True),
    ("fastapi", "fastapi", True),
    ("uvicorn", "uvicorn", True),
    ("python-multipart", "multipart", True),
    (
        "imageio-ffmpeg",
        "imageio_ffmpeg",
        False,
    ),  # opsional: video anotasi bisa diputar di browser
    ("pytest", "pytest", False),
    ("httpx", "httpx", False),
]
# Nama distribusi alternatif yang sah untuk satu modul.
ALT_DISTS = {
    "cv2": ["opencv-python", "opencv-python-headless", "opencv-contrib-python"],
    "lap": ["lap", "lapx"],
}


def version_of(package, module):
    for name in ALT_DISTS.get(module, [package]):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            continue
    return None


def load_specs():
    specs = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[.*\])?\s*(.*)$", line)
        if match and match.group(2):
            specs[match.group(1).lower().replace("_", "-")] = match.group(2)
    return specs


def satisfies(version, spec):
    try:
        from packaging.specifiers import SpecifierSet

        return version in SpecifierSet(spec)
    except Exception:
        return None  # tidak bisa dinilai


def main():
    print(f"Python  : {sys.version.split()[0]}  ({sys.executable})")
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        print("  ! proyek ditargetkan Python 3.10-3.13")
    specs, missing, off = load_specs(), [], []
    print("\nPaket:")
    for package, module, required in PACKAGES:
        version = version_of(package, module)
        try:
            importlib.import_module(module)
            importable = True
        except Exception as error:  # noqa: BLE001
            importable, reason = False, f"{type(error).__name__}: {error}"
        if version is None or not importable:
            tag = "KURANG" if required else "opsional, tidak ada"
            detail = (
                ""
                if importable or version is None
                else f" (terpasang {version} tapi gagal diimpor: {reason})"
            )
            print(f"  [{'X' if required else '-'}] {package:<17} {tag}{detail}")
            if required:
                missing.append(package)
            continue
        spec = specs.get(package.lower())
        ok = satisfies(version, spec) if spec else True
        note = "" if ok in (True, None) else f"  <- requirements.txt minta {spec}"
        print(
            f"  [{'OK' if ok in (True, None) else '!'}] {package:<17} {version}{note}"
        )
        if ok is False:
            off.append((package, version, spec))

    print("\nHardware:")
    try:
        import torch

        print(
            f"  CUDA tersedia: {torch.cuda.is_available()}"
            + (
                f" ({torch.cuda.get_device_name(0)})"
                if torch.cuda.is_available()
                else " (jalan di CPU)"
            )
        )
    except Exception:
        print("  torch belum bisa dipakai")

    print("\nFile & alat:")
    try:
        from detector_config import describe_detector, load_config

        info = describe_detector(load_config())
        print(
            f"  Detektor  : {info['label']} [{info['source']}] kelas={info['classes']} "
            f"-> {'siap' if info['ready'] else 'belum ada (akan diunduh/harus disiapkan)'}"
        )
    except Exception as error:  # noqa: BLE001
        print(f"  Detektor  : konfigurasi bermasalah: {error}")
    videomae = ROOT / "models" / "videomae-fall" / "model.safetensors"
    print(
        f"  VideoMAE  : {'ada lokal' if videomae.is_file() else 'belum ada lokal (diunduh dari HF saat sesi pertama, sekitar 345 MB)'}"
    )
    print(
        f"  node/npm  : {'ada' if shutil.which('node') and shutil.which('npm') else 'TIDAK ADA (perlu untuk build UI sekali)'}"
    )
    print(
        f"  frontend/dist: {'sudah dibuild' if (ROOT / 'frontend' / 'dist').is_dir() else 'belum dibuild'}"
    )
    print(
        f"  ffmpeg    : {'ada di PATH' if shutil.which('ffmpeg') else 'tidak di PATH (imageio-ffmpeg bisa menggantikan)'}"
    )

    print("\nRingkasan:")
    if missing:
        print("  Paket wajib yang kurang: " + ", ".join(missing))
        print(
            "  Pasang:  pip install "
            + " ".join(
                {
                    "opencv-python": "opencv-python",
                    "python-multipart": "python-multipart",
                }.get(m, m)
                for m in missing
            )
        )
    if off:
        print(
            "  Versi di luar requirements.txt (belum tentu masalah; coba jalan dulu):"
        )
        for package, version, spec in off:
            print(f"    {package} {version} (diminta {spec})")
    if not missing and not off:
        print("  Semua lengkap.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
