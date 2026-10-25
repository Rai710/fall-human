"""Siapkan model deteksi orang sesuai model_config.json tanpa menjalankan aplikasi.

Contoh:
  python download_detector.py                       # sesuai model_config.json
  python download_detector.py --source huggingface  # paksa sumber tertentu
"""
import argparse

from detector_config import SOURCES, describe_detector, load_config, resolve_detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, help="Timpa 'source' di model_config.json.")
    args = parser.parse_args()
    cfg = load_config()
    if args.source:
        cfg["source"] = args.source
    print("Konfigurasi:", describe_detector(cfg))
    result = resolve_detector(cfg, on_status=print)
    print(f"Siap: {result.path} (sumber: {result.source}, kelas: {result.classes})")
