"""Desktop controls for one-camera VideoMAE live replay and capture."""
from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import winsound

import cv2
from PIL import Image, ImageTk

from live_core import LiveConfig
from pose_engine import ROOT, process_source
from person_core import TrackingConfig
from person_engine import process_people


class PoseApp:
    def __init__(self, root):
        self.root = root
        root.title("Deteksi Jatuh | Uji Live VideoMAE")
        root.geometry("1160x940")
        root.minsize(900, 730)
        root.configure(bg="#10231f")
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self.closing = False
        self.pending_alerts = []
        self.last_output = ROOT / "outputs"
        self.last_frame = self.last_image = None
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#10231f")
        style.configure("TLabel", background="#10231f", foreground="#e6eee9", font=("Segoe UI", 10))
        style.configure("TButton", padding=(12, 8), font=("Segoe UI", 10))
        style.configure("TCheckbutton", background="#10231f", foreground="#e6eee9", font=("Segoe UI", 10))
        body = ttk.Frame(root, padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Uji kamera live dan rekaman CCTV", font=("Segoe UI", 23, "bold")).pack(anchor="w")
        ttk.Label(body, text="YOLO + ByteTrack → VideoMAE lokal per orang", foreground="#a8c8ba").pack(anchor="w", pady=(3, 12))

        alarm_bar = ttk.Frame(body)
        alarm_bar.pack(fill="x", pady=(0, 12))
        self.alarm_text = tk.StringVar(value="Tidak ada peringatan.")
        self.alarm_label = tk.Label(alarm_bar, textvariable=self.alarm_text, bg="#174437",
                                    fg="white", anchor="w", padx=10, pady=10,
                                    font=("Segoe UI", 11, "bold"), wraplength=790)
        self.alarm_label.pack(side="left", fill="x", expand=True)
        self.ack_button = ttk.Button(alarm_bar, text="Tandai ditangani", command=self.acknowledge, state="disabled")
        self.ack_button.pack(side="right", padx=(8, 0))

        controls = ttk.Frame(body)
        controls.pack(fill="x")
        self.video_button = ttk.Button(controls, text="Pilih video CCTV", command=self.choose_video)
        self.camera_button = ttk.Button(controls, text="Mulai webcam", command=self.start_camera)
        self.stream_button = ttk.Button(controls, text="CCTV / RTSP", command=self.start_stream)
        self.stop_button = ttk.Button(controls, text="Hentikan", command=self.stop_processing, state="disabled")
        for button in (self.video_button, self.camera_button, self.stream_button, self.stop_button):
            button.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Buka hasil", command=self.open_output).pack(side="right")

        source_bar = ttk.Frame(body)
        source_bar.pack(fill="x", pady=(12, 6))
        ttk.Label(source_bar, text="Nama kamera/lokasi:").pack(side="left")
        self.camera_name = tk.StringVar(value="Koridor 1")
        self.name_input = ttk.Entry(source_bar, textvariable=self.camera_name, width=25)
        self.name_input.pack(side="left", padx=(8, 20))
        ttk.Label(source_bar, text="Nomor webcam:").pack(side="left")
        self.camera_index = tk.StringVar(value="0")
        self.camera_input = ttk.Spinbox(source_bar, from_=0, to=9, textvariable=self.camera_index, width=3)
        self.camera_input.pack(side="left", padx=8)

        options = ttk.Frame(body)
        options.pack(fill="x", pady=(0, 6))
        self.realtime = tk.BooleanVar(value=True)
        self.save_video = tk.BooleanVar(value=True)
        self.allow_download = tk.BooleanVar(value=False)
        self.replay_check = ttk.Checkbutton(options, text="Simulasi live: ikuti waktu video", variable=self.realtime)
        self.save_check = ttk.Checkbutton(options, text="Simpan video beranotasi (file video)", variable=self.save_video)
        self.replay_check.pack(side="left", padx=(0, 16))
        self.save_check.pack(side="left")
        self.download_check = ttk.Checkbutton(body, text="Izinkan unduhan model pertama (sekitar 345 MB)",
                                             variable=self.allow_download)
        self.download_check.pack(anchor="w", pady=(0, 8))

        tracking_bar = ttk.Frame(body)
        tracking_bar.pack(fill="x", pady=(0, 8))
        self.per_person = tk.BooleanVar(value=True)
        self.person_check = ttk.Checkbutton(tracking_bar, text="Analisis per orang (YOLO + ByteTrack)", variable=self.per_person)
        self.person_check.pack(side="left", padx=(0, 12))
        self.detection_fps = tk.StringVar(value="10")
        self.max_people = tk.StringVar(value="6")
        ttk.Label(tracking_bar, text="Target FPS deteksi:").pack(side="left")
        self.detection_input = ttk.Spinbox(tracking_bar, from_=1, to=30, textvariable=self.detection_fps, width=4)
        self.detection_input.pack(side="left", padx=(5, 12))
        ttk.Label(tracking_bar, text="Batas orang dianalisis:").pack(side="left")
        self.people_input = ttk.Spinbox(tracking_bar, from_=1, to=16, textvariable=self.max_people, width=3)
        self.people_input.pack(side="left", padx=5)

        timing_bar = ttk.Frame(body)
        timing_bar.pack(fill="x", pady=(0, 10))
        self.interval = tk.StringVar(value="1.0")
        self.max_age = tk.StringVar(value="3.0")
        ttk.Label(timing_bar, text="Jeda analisis (dtk):").pack(side="left")
        self.interval_input = ttk.Spinbox(timing_bar, from_=0.5, to=10, increment=0.5, textvariable=self.interval, width=5)
        self.interval_input.pack(side="left", padx=(6, 18))
        ttk.Label(timing_bar, text="Batas usia hasil live (dtk):").pack(side="left")
        self.age_input = ttk.Spinbox(timing_bar, from_=1, to=30, increment=1, textvariable=self.max_age, width=5)
        self.age_input.pack(side="left", padx=(6, 18))
        ttk.Label(timing_bar, text="Klip 2 detik • 16 frame", foreground="#a8c8ba").pack(side="left")
        decision_bar = ttk.Frame(body)
        decision_bar.pack(fill="x", pady=(0, 10))
        self.threshold = tk.StringVar(value="75")
        self.confirm = tk.StringVar(value="1")
        ttk.Label(decision_bar, text="Ambang FallDown (%):").pack(side="left")
        self.threshold_input = ttk.Spinbox(decision_bar, from_=1, to=100, textvariable=self.threshold, width=5)
        self.threshold_input.pack(side="left", padx=(6, 18))
        ttk.Label(decision_bar, text="Konfirmasi hasil:").pack(side="left")
        self.confirm_input = ttk.Spinbox(decision_bar, from_=1, to=10, textvariable=self.confirm, width=4)
        self.confirm_input.pack(side="left", padx=(6, 18))
        ttk.Label(decision_bar, text="Ambang awal untuk percobaan; perlu dibandingkan pada rekaman berlabel.",
                  foreground="#a8c8ba").pack(side="left")

        self.canvas = tk.Canvas(body, bg="#081713", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.placeholder = self.canvas.create_text(0, 0, text="Pilih video atau mulai sumber kamera.",
                                                   fill="#aac2b6", font=("Segoe UI", 13))
        self.canvas.bind("<Configure>", self.redraw)
        self.prediction_text = tk.StringVar(value="Belum ada hasil analisis.")
        self.prediction_label = ttk.Label(body, textvariable=self.prediction_text, font=("Segoe UI", 11, "bold"))
        self.prediction_label.pack(anchor="w", pady=(10, 4))
        self.stats = tk.StringVar(value="Sumber belum dimulai.")
        ttk.Label(body, textvariable=self.stats).pack(anchor="w", pady=(0, 5))
        self.progress = ttk.Progressbar(body, maximum=100)
        self.progress.pack(fill="x", pady=(0, 7))
        self.status = tk.StringVar(value="Siap. Kamera aktif setelah tombol sumber ditekan. Unduhan model memerlukan persetujuan kotak di atas.")
        ttk.Label(body, textvariable=self.status, wraplength=1050).pack(anchor="w")
        ttk.Label(body, text="Alarm menunjukkan dugaan kejadian dalam kamera; petugas perlu memeriksa lokasi.",
                  foreground="#a8c8ba").pack(anchor="w", pady=(6, 0))
        self.config_widgets = (self.video_button, self.camera_button, self.stream_button,
            self.name_input, self.camera_input, self.replay_check, self.save_check,
            self.download_check, self.interval_input, self.age_input,
            self.threshold_input, self.confirm_input, self.person_check,
            self.detection_input, self.people_input)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(40, self.poll)
        root.after(4000, self.repeat_alarm)

    def sound_alarm(self):
        try:
            winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except RuntimeError:
            self.root.bell()

    def repeat_alarm(self):
        if not self.closing:
            if self.pending_alerts:
                self.sound_alarm()
            self.root.after(4000, self.repeat_alarm)

    def acknowledge(self):
        self.pending_alerts.clear()
        self.alarm_text.set("Peringatan ditandai ditangani. Riwayat tetap tersimpan.")
        self.alarm_label.configure(bg="#174437")
        self.ack_button.configure(state="disabled")

    def choose_video(self):
        path = filedialog.askopenfilename(title="Pilih rekaman CCTV", initialdir=str(ROOT / "data"),
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Semua file", "*.*")])
        if path:
            self.start(Path(path))

    def start_camera(self):
        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Webcam", "Nomor webcam harus 0 atau lebih.")
            return
        self.start(index)

    def start_stream(self):
        url = simpledialog.askstring("Alamat CCTV", "Masukkan URL RTSP/HTTP kamera (hanya dipakai selama sesi):", parent=self.root)
        if url:
            url = url.strip()
            if not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
                messagebox.showerror("Alamat CCTV", "Gunakan alamat RTSP atau HTTP kamera.")
                return
            self.start(url)

    def set_running(self, running):
        for widget in self.config_widgets:
            widget.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def start(self, source):
        if self.worker and self.worker.is_alive():
            return
        try:
            interval = float(self.interval.get().replace(",", "."))
            max_age = float(self.max_age.get().replace(",", "."))
            if interval < 0.5:
                raise ValueError("Jeda analisis minimal 0,5 detik.")
            config = LiveConfig(analysis_interval=interval, max_result_age=max_age,
                                fall_threshold=float(self.threshold.get().replace(",", ".")) / 100,
                                confirm_windows=int(self.confirm.get()))
            tracking = TrackingConfig(detection_fps=float(self.detection_fps.get().replace(",", ".")),
                                      max_people=int(self.max_people.get()))
        except ValueError as error:
            messagebox.showerror("Pengaturan waktu", str(error))
            return
        self.stop.clear()
        self.pending_alerts.clear()
        self.alarm_text.set("Tidak ada peringatan.")
        self.alarm_label.configure(bg="#174437")
        self.ack_button.configure(state="disabled")
        for channel in (self.frames, self.events):
            while True:
                try:
                    channel.get_nowait()
                except queue.Empty:
                    break
        self.set_running(True)
        self.last_frame = None
        self.canvas.delete("all")
        self.placeholder = self.canvas.create_text(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2,
            text="Menyiapkan model dan sumber...", fill="#aac2b6", font=("Segoe UI", 13))
        self.progress["value"] = 0
        self.prediction_text.set("Belum ada hasil analisis.")
        self.status.set("Memuat model lokal. Persiapan pertama dapat memerlukan waktu.")
        options = dict(save_video=self.save_video.get(), realtime=self.realtime.get(),
                       local_only=not self.allow_download.get(), camera_name=self.camera_name.get(), config=config)
        engine = process_people if self.per_person.get() else process_source
        if self.per_person.get():
            options["tracking"] = tracking

        def on_frame(frame, stats):
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait((frame, stats))

        def on_event(kind, payload):
            # Alarms do not travel in the preview queue, which may skip frames.
            self.events.put((kind, payload))

        def work():
            try:
                result = engine(source, stop=self.stop, on_frame=on_frame,
                                        on_event=on_event, **options)
                self.events.put(("done", result))
            except Exception as error:
                self.events.put(("error", str(error)))

        self.worker = threading.Thread(target=work, name="live-session", daemon=False)
        self.worker.start()

    def render(self, frame):
        self.last_frame = frame
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        picture.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.last_image = ImageTk.PhotoImage(picture)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self.last_image)

    def redraw(self, _event=None):
        if self.last_frame is not None:
            self.render(self.last_frame)
        else:
            self.canvas.coords(self.placeholder, self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2)

    def poll(self):
        try:
            frame, stats = self.frames.get_nowait()
            self.render(frame)
            label = stats["label"] or f"Mengumpulkan klip {stats['buffer_frames']}/{stats['buffer_target']} frame"
            analysis = "—" if stats["analysis_ms"] is None else f"{stats['analysis_ms']/1000:.2f} dtk"
            age = "—" if stats["result_age_ms"] is None else f"{stats['result_age_ms']/1000:.2f} dtk"
            prefix = "HASIL TERLAMBAT | " if stats["stale"] else ""
            self.prediction_text.set(f"{prefix}{label} | analisis {analysis} | usia hasil {age}")
            self.prediction_label.configure(foreground="#ffbf69" if stats["stale"] else "#e6eee9")
            self.stats.set(f"{stats['mode']} | waktu sumber {stats['timestamp_ms']/1000:.1f} dtk | "
                f"baca {stats['capture_fps']:.1f} FPS | {stats['analyses']} analisis | "
                f"{stats['replaced_clips']} klip antrean diganti | {stats['alerts_count']} alarm")
            if "tracked_people" in stats:
                detector = "—" if stats["detector_ms"] is None else f"{stats['detector_ms']:.0f} ms"
                self.stats.set(f"{stats['mode']} | baca {stats['capture_fps']:.1f} FPS | "
                    f"tracking {stats['tracking_fps']:.1f} FPS ({detector}) | "
                    f"{stats['tracked_people']} ID | {stats['analyses']} analisis | "
                    f"{stats['pending_people']} menunggu | {stats['alerts_count']} alarm")
                if stats["overflow_people"]:
                    self.stats.set(self.stats.get() + f" | {stats['overflow_people']} orang DI LUAR BATAS")
                if stats["tracking_stale"]:
                    self.stats.set(self.stats.get() + " | TRACKING TERLAMBAT")
            if stats["total_frames"]:
                self.progress["value"] = 100 * stats["frame"] / stats["total_frames"]
        except queue.Empty:
            pass
        try:
            while True:
                event, result = self.events.get_nowait()
                if event == "status":
                    if not self.stop.is_set():
                        self.status.set(result)
                elif event == "alarm":
                    self.pending_alerts.append(result)
                    start, end = result["clip_start_ms"] / 1000, result["clip_end_ms"] / 1000
                    identity = f" | ID {result['track_id']}.{result['segment_id']}" if "track_id" in result else ""
                    self.alarm_text.set(f"ALARM {result['camera_name']}{identity}: diduga jatuh | klip {start:.1f}–{end:.1f} dtk "
                                        f"| kejadian #{result['alert_id']}")
                    self.alarm_label.configure(bg="#a82121")
                    self.ack_button.configure(state="normal")
                    self.sound_alarm()
                elif event == "done":
                    self.set_running(False)
                    self.last_output, summary = result
                    label = "Selesai" if summary["status"] == "completed" else "Dihentikan"
                    seconds = summary.get("analysis_mean_ms")
                    average = "belum ada analisis" if seconds is None else f"rata-rata {seconds/1000:.2f} dtk/analisis"
                    text = f"{label}. {summary['alerts_count']} alarm; {average}. Hasil: {self.last_output.name}."
                    if summary["warnings"]:
                        text += " " + " ".join(summary["warnings"])
                    self.status.set(text)
                elif event == "error":
                    self.set_running(False)
                    self.status.set("Sesi gagal. Ringkasan kegagalan tersimpan di folder outputs.")
                    if not self.closing:
                        messagebox.showerror("Sesi gagal", result)
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(40, self.poll)

    def stop_processing(self):
        self.stop.set()
        self.status.set("Menghentikan sumber. Menunggu analisis aktif selesai dan menyimpan hasil...")
        self.stop_button.configure(state="disabled")

    def open_output(self):
        self.last_output.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.last_output))

    def on_close(self):
        self.closing = True
        self.stop_processing()


if __name__ == "__main__":
    window = tk.Tk()
    PoseApp(window)
    window.mainloop()
