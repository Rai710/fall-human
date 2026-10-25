# Deteksi orang jatuh — React + FastAPI + YOLO + VideoMAE

Dashboard ini menggabungkan **YoloDetector.pt** milik tim detector dengan
**VideoMAE Le2i + CAUCAFall dua kelas** (`no_fall`, `fall`). Integrasi dikerjakan
pada source `fall-human-merged` di checkout branch `final`.

## Menjalankan di komputer ini

Klik **run_dashboard.cmd**, lalu buka **http://localhost:8000**.
Launcher memakai `.venv-yolo` jika tersedia, atau interpreter lokal yang
dikonfigurasi dalam `local_python.txt` (diabaikan Git). Build React disediakan
di `frontend/dist`. Kamera belum dibuka saat dashboard baru diakses.

Untuk instalasi mandiri pada komputer lain, jalankan `setup_dashboard.cmd`, lalu
`run_dashboard.cmd`. Python 3.10–3.13 dan Node/npm diperlukan. Bobot model tidak
ikut Git; salin kedua paket model secara terpisah.

## Dua mode analisis

1. **Seluruh kamera** (default): VideoMAE membaca adegan penuh dengan cara input
   yang sesuai training. Mode ini tidak menjalankan YOLO/ByteTrack dan tidak
   mengidentifikasi siapa yang jatuh.
2. **Per orang** (eksperimen): YOLO → ByteTrack → klip RGB/crop per ID → VideoMAE.
   Bobot classifier yang sama dipakai bergantian untuk klip setiap orang.
   Crop mengubah input dari training seluruh adegan dan memerlukan evaluasi
   akurasi tersendiri. Area terakhir saat track hilang diberi penanda khusus.

Mulai dengan **Upload video**; simulasi live pada file mati secara default.
Mode offline menunggu analisis klip sehingga waktu proses dapat melebihi durasi
video. Webcam selalu mengikuti penjadwalan live, walaupun pilihan replay file
tidak aktif. Webcam index menunjuk kamera mesin backend, bukan browser pengguna.

## Bobot yang diperlukan

```text
models/
├── YoloDetector.pt
└── best_model-20261025T045257Z-1-001/
    └── best_model/
        ├── model.safetensors
        ├── config.json
        ├── preprocessor_config.json
        ├── inference_config.json
        ├── export_complete.json
        └── ... seluruh berkas ekspor lainnya
```

YOLO bertipe `detect`, kelas **0 = person**. `model_config.json` menunjuk
`models/YoloDetector.pt`; opsi lokal/pretrained/Hugging Face tetap tersedia.

Classifier memakai epoch 4 (`mixed_fine_tuning`), input 16 gambar RGB sekitar
2 detik, letterbox hitam PIL bilinear 224×224, normalisasi processor tersimpan,
stride 0,5 detik dan satu konfirmasi awal. Threshold ekspor tetap **0,58**;
ambang alarm aplikasi dinaikkan menjadi **0,70 (70%)** di `action_config.py`. Dua hasil
di bawah ambang memulihkan gate untuk kejadian berikutnya. Label positif mencakup
gerakan jatuh atau keadaan setelah jatuh. Skor belum terkalibrasi sebagai peluang.

Dashboard memakai paket lokal secara eksplisit. Tidak ada unduhan atau fallback
otomatis ke model HF tujuh kelas ketika classifier ini tidak lengkap. Loader
memeriksa hash ekspor; pertahankan seluruh isi `best_model` tanpa mengubahnya.

## Bagian aplikasi

- `frontend/`: React/TypeScript/Vite; upload, progres, alarm, review, dan riwayat.
- `backend/`: FastAPI; sesi, preview MJPEG, event WebSocket, laporan dan video.
- `action_config.py`: lokasi/metadata classifier dua kelas.
- `mixed_model_config.py`: validasi ekspor dan preprocessing sesuai training.
- `pose_engine.py`: loader classifier dan analisis seluruh kamera.
- `person_engine.py`: YOLO/ByteTrack, crop per orang, dan pengelolaan track hilang.
- `live_core.py`, `person_core.py`: buffer, antrean, dan gate alarm.

API `POST /api/sessions/start` menerima file atau index webcam, `save_video`,
`realtime` (file default false), dan `analysis_scope` (`scene` atau `person`,
default `scene`). `GET /api/config` menjelaskan detector dan classifier aktif.
Satu sesi aktif diizinkan per proses backend; permintaan lain ditolak HTTP 409,
bukan dimasukkan ke antrean pengguna. Jangan jalankan beberapa worker Uvicorn
untuk berbagi model/session state pada implementasi ini.

Output di `outputs/<sesi>/`: video H.264 bila konversi berhasil, prediksi/alarm
CSV, summary JSON, preview, dan review. Riwayat lama tetap bisa dibaca.

## Pengembangan dan pengujian

Backend: `python -m uvicorn backend.main:app --port 8000`.
Frontend: `npm --prefix frontend ci`, lalu `npm --prefix frontend run dev`.
Frontend dev mem-proxy `/api` dan WebSocket ke port 8000. Build: `npm --prefix frontend run build`.

Tes otomatis: `python -m pytest -q`. Tes API/engine memakai model palsu;
tes tersebut tidak mengukur akurasi. Uji kedua model asli lewat API:

```powershell
python tests/smoke_real_models.py "D:\video\contoh.mp4"
```

Lihat **INTEGRASI_MODEL.md** untuk perbandingan sebelum/sesudah dan hasil uji aktual.
`README_ENGINE.md` dan `VERIFIKASI.md` adalah dokumentasi engine versi sebelumnya.
Aplikasi desktop `MULAI.cmd` tetap merupakan entry point lama; gunakan dashboard
untuk konfigurasi gabungan terbaru ini.

## Batas integrasi

Model terbaru pernah salah menandai orang rebah di kasur sebagai fall. Integrasi
tidak memperbaiki akurasi bobot atau membuktikan performa CCTV multi-orang.
Stride 0,5 detik tidak menjamin inferensi selesai setiap 0,5 detik. CPU dapat
tertinggal pada mode live; crop per orang perlu dievaluasi secara terpisah.

Source ini disiapkan untuk lokal/satu origin. Deployment frontend dan backend
terpisah masih memerlukan konfigurasi URL API/WebSocket, CORS, kontrol akses,
dan penyimpanan hasil yang sesuai. Belum ada deployment cloud dalam pekerjaan ini.

Checkpoint asal VideoMAE mencantumkan CC BY-NC 4.0; Ultralytics memakai AGPL-3.0
atau lisensi komersial yang sesuai. Ketentuan bobot YOLO custom perlu mengikuti
asal training dan izin tim pembuatnya.
