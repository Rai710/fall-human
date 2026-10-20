## RGB ZIP -> MP4

import argparse
import re
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def convert(zip_path: Path, out_dir: Path, fps: float) -> Path:
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(
            (n for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))),
            key=natural_key,
        )
        if not names:
            raise RuntimeError("tidak ada gambar PNG/JPG di dalam zip")
        out_path = out_dir / (zip_path.stem + ".mp4")
        writer, size = None, None
        for i, name in enumerate(names):
            frame = cv2.imdecode(
                np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                raise RuntimeError(f"gambar rusak: {name}")
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
                )
                if not writer.isOpened():
                    raise RuntimeError("codec mp4v tidak tersedia untuk menulis MP4")
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
        writer.release()
    print(
        f"{zip_path.name}: {len(names)} frame, {size[0]}x{size[1]}, {fps:g} fps "
        f"(~{len(names) / fps:.1f} dtk) -> {out_path}"
    )
    return out_path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("zips", nargs="+", type=Path)
    p.add_argument(
        "--out", type=Path, help="folder hasil (awal: folder zip masing-masing)"
    )
    p.add_argument(
        "--fps", type=float, default=30.0, help="frame per detik video hasil (awal: 30)"
    )
    args = p.parse_args()
    if not 1 <= args.fps <= 120:
        p.error("--fps harus 1-120")
    failed = 0
    for z in args.zips:
        try:
            out_dir = args.out or z.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            convert(z, out_dir, args.fps)
        except Exception as error:
            failed += 1
            print(f"GAGAL {z.name}: {error}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
