"""Headless camera-level VideoMAE analysis (the historical filename is retained)."""
import argparse
import json
from pathlib import Path
import threading

from live_core import LiveConfig
from pose_engine import ROOT, process_source
from person_core import TrackingConfig
from person_engine import process_people


def main():
    parser = argparse.ArgumentParser(description="Analisis VideoMAE per kamera: simulasi live atau offline.")
    parser.add_argument("source", nargs="?", help="File video atau alamat RTSP/HTTP kamera.")
    parser.add_argument("--camera", type=int, help="Nomor webcam; gunakan sebagai pengganti source.")
    parser.add_argument("--offline", action="store_true", help="Analisis semua klip terjadwal pada file, tanpa mengikuti waktu video.")
    parser.add_argument("--name", default="Kamera 1", help="Nama lokasi yang akan muncul pada alarm.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model-dir", type=Path, help="Folder model VideoMAE yang sudah lengkap.")
    parser.add_argument("--allow-download", action="store_true", help="Izinkan unduhan model dari Hugging Face jika diperlukan.")
    parser.add_argument("--no-video", action="store_true", help="Tidak menyimpan video beranotasi.")
    parser.add_argument("--interval", type=float, default=1.0, help="Jeda pengajuan klip baru dalam detik.")
    parser.add_argument("--window", type=float, default=2.0, help="Rentang waktu yang diwakili 16 frame.")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--confirm", type=int, default=1)
    parser.add_argument("--max-age", type=float, default=3.0, help="Hasil live lebih tua dari ini tidak mengonfirmasi alarm.")
    parser.add_argument("--scene", action="store_true", help="Bandingkan dengan backend seluruh kamera tanpa tracking.")
    parser.add_argument("--detector", type=Path, help="Path file YOLO (menimpa model_config.json); default: sesuai model_config.json.")
    parser.add_argument("--detection-fps", type=float, default=10.0)
    parser.add_argument("--max-people", type=int, default=6)
    args = parser.parse_args()
    if (args.source is None) == (args.camera is None):
        parser.error("Pilih satu: source atau --camera.")
    if args.camera is not None and args.camera < 0:
        parser.error("Nomor webcam tidak boleh negatif.")
    try:
        config = LiveConfig(window_seconds=args.window, analysis_interval=args.interval,
                            fall_threshold=args.threshold, confirm_windows=args.confirm,
                            max_result_age=args.max_age)
        tracking = TrackingConfig(detection_fps=args.detection_fps, max_people=args.max_people)
    except ValueError as error:
        parser.error(str(error))
    stop = threading.Event()
    try:
        engine = process_source if args.scene else process_people
        extra = {} if args.scene else dict(tracking=tracking, detector_path=args.detector)
        folder, summary = engine(
            args.camera if args.camera is not None else args.source,
            save_video=not args.no_video, stop=stop, realtime=not args.offline,
            config=config, camera_name=args.name, output_root=args.output,
            model_path=args.model_dir, local_only=not args.allow_download,
            on_event=lambda kind, value: print(f"[{kind}] {value}"), **extra)
    except KeyboardInterrupt:
        parser.exit(130, "\nDihentikan. Ringkasan sesi disimpan di folder keluaran.\n")
    except (RuntimeError, OSError, ValueError) as error:
        parser.exit(1, f"Gagal: {error}\n")
    print(json.dumps({"output": str(folder), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
