"""Headless extraction, useful for batch dataset processing."""
import argparse
import json
from pathlib import Path

from pose_engine import DEFAULT_MODEL, ROOT, process_source


def main():
    parser = argparse.ArgumentParser(description="Ekstrak 17 titik tubuh per orang dengan YOLO Pose dan ByteTrack.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--no-video", action="store_true", help="Hanya ekspor CSV dan ringkasan.")
    parser.add_argument("--max-frames", type=int, help="Batas frame untuk percobaan cepat.")
    parser.add_argument("--max-people", type=int, default=10, help="Batas deteksi orang per frame.")
    parser.add_argument("--image-size", type=int, default=640, help="Ukuran masukan model; lebih tinggi memerlukan lebih banyak waktu.")
    parser.add_argument("--device", default="cpu", help="cpu atau indeks GPU jika PyTorch CUDA telah dipasang.")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames harus lebih dari nol")
    if not 1 <= args.max_people <= 100 or args.image_size < 32:
        parser.error("Batas orang harus 1–100 dan ukuran gambar setidaknya 32.")
    try:
        session, summary = process_source(args.video, args.output, args.model,
                                          save_video=not args.no_video, max_frames=args.max_frames,
                                          max_people=args.max_people, image_size=args.image_size,
                                          device=args.device)
    except (RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f"Gagal: {error}\n")
    print(json.dumps({"output": str(session), **summary}, indent=2))


if __name__ == "__main__":
    main()
