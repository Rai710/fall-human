"""Local desktop preview for the hospital fall-detection pose prototype."""
from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import winsound

import cv2
from PIL import Image, ImageTk

from pose_engine import ROOT, process_source


class PoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pemantauan Jatuh | Beberapa Orang")
        self.root.geometry("1080x800")
        self.root.minsize(800, 620)
        self.root.configure(bg="#10231f")
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.stop = threading.Event()
        self.worker = None
        self.closing = False
        self.last_output = ROOT / "outputs"
        self.last_image = None
        self.last_frame = None
        self.pending_alerts = []
        self.style = ttk.Style(root)
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#10231f")
        self.style.configure("TLabel", background="#10231f", foreground="#e6eee9", font=("Segoe UI", 11))
        self.style.configure("TButton", padding=(16, 10), font=("Segoe UI", 10))
        self.style.configure("TCheckbutton", background="#10231f", foreground="#e6eee9", font=("Segoe UI", 10))
        body = ttk.Frame(root, padding=22)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Pantau gerakan dan tanda orang jatuh.",
                  font=("Segoe UI", 23, "bold")).pack(anchor="w")
        ttk.Label(body, text="Beberapa orang sekaligus  |  Kerangka dan nomor ID sementara",
                  foreground="#a8c8ba").pack(anchor="w", pady=(4, 16))
        alarm_bar = ttk.Frame(body)
        alarm_bar.pack(fill="x", pady=(0, 12))
        self.alarm_text = tk.StringVar(value="Tidak ada peringatan jatuh.")
        self.alarm_label = tk.Label(alarm_bar, textvariable=self.alarm_text,
                                    bg="#174437", fg="#e6eee9", anchor="w",
                                    padx=12, pady=12, font=("Segoe UI", 12, "bold"))
        self.alarm_label.pack(side="left", fill="x", expand=True)
        self.ack_button = ttk.Button(alarm_bar, text="Tandai ditangani", command=self.acknowledge,
                                     state="disabled")
        self.ack_button.pack(side="right", padx=(8, 0))
        controls = ttk.Frame(body)
        controls.pack(fill="x")
        self.video_button = ttk.Button(controls, text="Pilih & proses video", command=self.choose_video)
        self.video_button.pack(side="left", padx=(0, 8))
        self.camera_button = ttk.Button(controls, text="Mulai webcam", command=self.start_camera)
        self.camera_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(controls, text="Hentikan", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Button(controls, text="Buka hasil", command=self.open_output).pack(side="right")
        options = ttk.Frame(body)
        options.pack(fill="x", pady=(10, 12))
        self.save_video = tk.BooleanVar(value=True)
        self.save_check = ttk.Checkbutton(options, text="Simpan video dengan kerangka (untuk file video)",
                                         variable=self.save_video)
        self.save_check.pack(side="left")
        camera_options = ttk.Frame(options)
        camera_options.pack(side="right")
        ttk.Label(camera_options, text="Webcam:").pack(side="left", padx=(12, 4))
        self.camera_index = tk.StringVar(value="0")
        self.camera_input = ttk.Spinbox(camera_options, from_=0, to=9, textvariable=self.camera_index, width=3)
        self.camera_input.pack(side="left")
        limits = ttk.Frame(body)
        limits.pack(fill="x", pady=(0, 10))
        ttk.Label(limits, text="Batas orang dalam gambar:").pack(side="left", padx=(0, 8))
        self.max_people = tk.StringVar(value="10")
        self.people_input = ttk.Spinbox(limits, from_=1, to=50, textvariable=self.max_people, width=4)
        self.people_input.pack(side="left")
        ttk.Label(limits, text="ID membantu mengikuti orang dalam sesi ini; dapat berubah saat orang tertutup.",
                  foreground="#a8c8ba").pack(side="left", padx=12)
        self.canvas = tk.Canvas(body, bg="#081713", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.placeholder = self.canvas.create_text(0, 0, fill="#aac2b6", font=("Segoe UI", 14),
                                                  text="Pilih video untuk melihat kerangka tubuh.", tags="placeholder")
        self.canvas.bind("<Configure>", self.redraw)
        self.stats = tk.StringVar(value="Belum ada video diproses.")
        ttk.Label(body, textvariable=self.stats, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(12, 5))
        self.progress = ttk.Progressbar(body, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 9))
        self.status = tk.StringVar(value="Siap. Webcam baru aktif setelah kamu menekan Mulai webcam.")
        ttk.Label(body, textvariable=self.status, wraplength=1000).pack(anchor="w")
        ttk.Label(body, text="Alarm adalah dugaan jatuh. Petugas tetap perlu memeriksa orang dan lokasi.",
                  foreground="#a8c8ba").pack(anchor="w", pady=(8, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(40, self.poll)
        self.root.after(4000, self.repeat_alarm)

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
        self.alarm_text.set("Peringatan sudah ditandai ditangani. Riwayat tetap tersimpan.")
        self.alarm_label.configure(bg="#174437")
        self.ack_button.configure(state="disabled")

    def choose_video(self):
        path = filedialog.askopenfilename(title="Pilih video", initialdir=str(ROOT / "data"),
                                          filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Semua file", "*.*")])
        if path:
            self.start(Path(path))

    def start_camera(self):
        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Nomor webcam", "Masukkan nomor webcam 0 atau lebih.")
            return
        self.start(index)

    def set_running(self, running):
        for widget in (self.video_button, self.camera_button, self.save_check, self.camera_input, self.people_input):
            widget.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def start(self, source):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            max_people = int(self.max_people.get())
            if not 1 <= max_people <= 50:
                raise ValueError
        except ValueError:
            messagebox.showerror("Batas orang", "Masukkan batas antara 1 dan 50 orang.")
            return
        self.stop.clear()
        self.pending_alerts.clear()
        self.alarm_text.set("Tidak ada peringatan jatuh.")
        self.alarm_label.configure(bg="#174437")
        self.ack_button.configure(state="disabled")
        while not self.frames.empty():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
        self.set_running(True)
        self.last_frame = None
        self.canvas.delete("all")
        self.placeholder = self.canvas.create_text(self.canvas.winfo_width() // 2,
                                                  self.canvas.winfo_height() // 2,
                                                  text="Membuka sumber dan memuat model...",
                                                  fill="#aac2b6", font=("Segoe UI", 14), tags="placeholder")
        self.stats.set("Memuat model...")
        self.progress["value"] = 0
        self.status.set("CSV tersimpan selama pemrosesan. Hentikan untuk menyelesaikan sesi.")
        export = self.save_video.get()

        def on_frame(frame, stats):
            # Alarms have their own unbounded queue and cannot be lost with a preview frame.
            for alarm in stats.get("new_alerts", []):
                self.events.put(("alarm", alarm))
            # Keep the latest preview only. The engine still writes every frame's data.
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait((frame.copy(), stats))

        def work():
            try:
                result = process_source(source, save_video=export, stop=self.stop,
                                        on_frame=on_frame, realtime=True, max_people=max_people)
                self.events.put(("done", result))
            except Exception as error:
                self.events.put(("error", str(error)))

        self.worker = threading.Thread(target=work, name="pose-extraction", daemon=False)
        self.worker.start()

    def render(self, frame):
        self.last_frame = frame
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.last_image = ImageTk.PhotoImage(image)
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
            pose = f"{stats['people']} orang dengan ID | {stats['usable_poses']} pose cukup jelas"
            self.stats.set(f"{pose}   |   Frame {stats['frame']}   |   "
                           f"Waktu {stats['timestamp_ms'] / 1000:.1f} dtk   |   "
                           f"Alarm {stats['alerts_count']}   |   Inferensi {stats['inference_fps']:.1f} FPS")
            if stats["total_frames"]:
                self.progress["value"] = 100 * stats["frame"] / stats["total_frames"]
        except queue.Empty:
            pass
        try:
            while True:
                event, result = self.events.get_nowait()
                if event == "alarm":
                    self.pending_alerts.append(result)
                    ids = ", ".join(str(alarm["track_id"]) for alarm in self.pending_alerts)
                    seconds = result["timestamp_ms"] / 1000
                    self.alarm_text.set(f"ALARM: diduga jatuh - ID {ids} (detik {seconds:.1f}). Periksa segera.")
                    self.alarm_label.configure(bg="#a82121")
                    self.ack_button.configure(state="normal")
                    self.sound_alarm()
                    continue
                self.set_running(False)
                if event == "done":
                    self.last_output, summary = result
                    label = "Selesai" if summary["status"] == "completed" else "Dihentikan"
                    self.status.set(f"{label}. {summary['alerts_count']} alarm; "
                                    f"maksimal {summary['max_simultaneous_tracked']} orang bersamaan; "
                                    f"{summary['unique_track_ids']} ID selama sesi. Hasil: {self.last_output.name}")
                    if summary.get("warning"):
                        self.status.set(self.status.get() + " | " + summary["warning"])
                else:
                    self.status.set("Gagal. Periksa pesan kesalahan dan coba lagi.")
                    if not self.closing:
                        messagebox.showerror("Pemrosesan gagal", result)
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(40, self.poll)

    def stop_processing(self):
        self.stop.set()
        self.status.set("Menyelesaikan dan menyimpan hasil...")
        self.stop_button.configure(state="disabled")

    def open_output(self):
        self.last_output.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.last_output))

    def on_close(self):
        self.closing = True
        self.stop_processing()


if __name__ == "__main__":
    root = tk.Tk()
    PoseApp(root)
    root.mainloop()
