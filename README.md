# Deteksi dugaan orang jatuh — VideoMAE + kerangka MediaPipe

Aplikasi desktop untuk memantau video atau webcam dan memberi **alarm dugaan jatuh**.

- **Yang memutuskan jatuh atau tidak:** model VideoMAE dari Hugging Face
  ([`yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect`](https://huggingface.co/yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect)).
  Model ini mengenali 7 aktivitas: `FallDown`, `LyingDown`, `SitDown`, `Sitting`, `StandUp`, `Standing`, `Walking`.
- **MediaPipe hanya menggambar kerangka tubuh** di video sebagai tampilan. Hasilnya tidak ikut
  menentukan alarm. Jika MediaPipe atau modelnya tidak tersedia, deteksi jatuh tetap berjalan tanpa kerangka.
- **GPU NVIDIA dipakai otomatis** bila ada (fp16). Tanpa GPU, aplikasi berjalan di CPU.

**Alarm berarti dugaan jatuh, bukan kepastian medis.** Petugas tetap perlu memeriksa orang dan lokasinya.

## Menjalankan

Syarat: **Windows 10/11**, **Python 3.10–3.13 dari python.org** (centang *tcl/tk and IDLE* saat
memasang; disarankan 3.12), dan internet untuk penggunaan pertama.

1. Klik dua kali **`MULAI.cmd`**.
2. Penggunaan pertama memasang lingkungan Python, paket (PyTorch berukuran besar), dan mengunduh
   model. Jika ada GPU NVIDIA, PyTorch versi CUDA dipasang otomatis. Setelah itu startnya cepat.
3. Pilih **Pilih & proses video** atau **Mulai webcam**. **Hentikan** menyelesaikan sesi,
   **Buka hasil** membuka folder hasil, **Tandai ditangani** mematikan bunyi alarm.

Di jendela konsol akan tertulis `Model berjalan di GPU: ...` atau `Model berjalan di CPU.`

## Cara keputusan dibuat

Video diambil sekitar 7,5 frame per detik; tiap 16 frame terakhir (sekitar 2 detik) dianalisis
VideoMAE sekitar tiap 0,27 detik. Alarm berbunyi bila:

- kelas `FallDown` menang dengan keyakinan **> 75%**, dan
- itu terjadi pada **2 analisis berturut-turut**, dan
- sudah lewat **10 detik** sejak alarm sebelumnya (supaya orang yang masih terbaring tidak memicu alarm berulang).

Angka-angka itu ada di bagian atas `pose_engine.py` (`FALL_THRESHOLD`, `CONFIRM_WINDOWS`, `ALERT_COOLDOWN_S`).
`USE_SKELETON = False` mematikan kerangka sepenuhnya.

## Keluaran

Tiap sesi membuat folder `outputs/<waktu>_<nama>/`:

- `annotated.mp4` — video dengan kerangka dan status (hanya untuk file video; audio tidak disalin)
- `predictions.csv` — hasil tiap analisis: label, keyakinan, apakah dihitung jatuh
- `alerts.csv` — waktu dan keyakinan tiap alarm
- `summary.json` — ringkasan sesi (perangkat, jumlah analisis, alarm, peringatan)

## Hal yang perlu diketahui

- **Tidak ada ID per orang.** VideoMAE menilai seluruh adegan. Angka "ID" pada banner alarm adalah
  **nomor kejadian jatuh** (1, 2, 3, ...). Kerangka digambar per orang (batas jumlahnya diatur lewat
  *Batas orang dalam gambar*), tetapi tanpa nomor.
- Pada CPU yang lambat, label status bisa terlambat sekitar selama satu analisis; waktu alarm di CSV
  tetap waktu video yang sebenarnya dianalisis.
- Prototipe ini belum dievaluasi pada rekaman CCTV rumah sakit. Uji dengan video dari lokasi
  sesungguhnya sebelum dipercaya.
- **Lisensi model: CC BY-NC 4.0 (non-komersial).** Periksa sebelum memakainya di luar penelitian/pembelajaran.
- Aplikasi hanya untuk Windows (memakai `winsound` dan `os.startfile`).
- Jika memakai dataset GMDCSA-24, lihat atribusi di `data/README.md`.

## Pemecahan masalah

| Gejala | Yang dilakukan |
| --- | --- |
| Konsol menulis `Model berjalan di CPU` padahal ada GPU NVIDIA | Perbarui driver NVIDIA, hapus file `.cuda_checked`, lalu jalankan `MULAI.cmd` lagi (pemasangan PyTorch CUDA diulang). |
| Kerangka tidak muncul, status bar menulis "Kerangka MediaPipe nonaktif" | Biasanya model kerangka belum terunduh. Pastikan internet aktif lalu jalankan ulang; atau unduh manual `pose_landmarker_lite.task` ke folder `models/`. Deteksi jatuh tetap jalan tanpa kerangka. |
| Galat soal `cv2` / dua paket OpenCV | Jalankan `.venv\Scripts\python -m pip uninstall -y opencv-python` (atau `.venv-yolo\...`), lalu `MULAI.cmd` lagi. |
| Ingin mengulang pemasangan dari awal | Hapus folder `.venv` (atau `.venv-yolo`), lalu jalankan `MULAI.cmd`. |

## Uji otomatis

`python tests/test_engine.py` menguji logika engine (alarm, jeda, tombol Hentikan, error, kerangka)
dengan model tiruan, tanpa mengunduh apa pun dan tanpa GPU.

## Isi folder

| File | Fungsi |
| --- | --- |
| `MULAI.cmd`, `start.ps1` | Pemasangan otomatis dan peluncur |
| `app.py` | Antarmuka desktop (Tkinter) |
| `pose_engine.py` | Mesin deteksi: VideoMAE, kerangka MediaPipe, alarm, ekspor |
| `download_model.py` | Mengunduh model dan menyimpan nama/lokasi model |
| `requirements.txt` | Daftar paket |
| `tests/test_engine.py` | Uji engine |
| `data/` | Tempat video percobaan (tidak dimasukkan ke Git) |
