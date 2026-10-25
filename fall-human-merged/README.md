# Panduan teknis aplikasi

Panduan fitur, instalasi, penempatan model dan batas penggunaan tersedia di
[README utama repository](../README.md). Rincian kerja tim tersedia di
[PEMBAGIAN_TUGAS.md](../PEMBAGIAN_TUGAS.md).

## Tim dan pembagian tugas

| Nama | Fokus |
|---|---|
| **Raihan Buono** | VideoMAE, preprocessing, threshold, dan evaluasi model jatuh |
| **Michael** | YOLO detector orang, ByteTrack, dan evaluasi tracking |
| **Rioo** | Frontend React, tampilan dashboard, dan interaksi pengguna |

Integrasi backend/API, pengujian sistem, dan laporan dikerjakan bersama.

Semua perintah di halaman ini dijalankan dari folder **`fall-human-merged`**.

## Menjalankan di Windows

```powershell
.\setup_dashboard.cmd
.\run_dashboard.cmd
```

Setup pertama membuat `.venv-yolo` dan memasang paket. Launcher membangun React
jika hasil build belum ada, lalu menyajikan frontend dan backend pada
**http://localhost:8000**. Bila sudah memiliki runtime lokal yang cocok, launcher
dapat membacanya dari `local_python.txt`; file ini tidak ikut Git.

Dashboard memakai paket VideoMAE lokal. Jika pesan akhir script setup lama
menyebut unduhan model otomatis, yang berlaku untuk dashboard gabungan adalah
konfigurasi lokal di `action_config.py`; paket model tetap harus disiapkan.

## Mode pengembangan

Terminal backend:

```powershell
.\.venv-yolo\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

Terminal frontend:

```powershell
npm --prefix frontend ci
npm --prefix frontend run dev
```

Buka alamat yang dicetak Vite. Konfigurasi dev mem-proxy `/api` dan WebSocket ke
port 8000. Untuk membuat build baru:

```powershell
npm --prefix frontend run build
```

Jika memakai `local_python.txt`, ganti path interpreter pada perintah Python
dengan interpreter yang tercatat di file tersebut.

## Konfigurasi model

| File | Peran |
|---|---|
| `model_config.json` | Sumber bobot YOLO, `person_class_ids`, confidence detector |
| `tracker.yaml` | Ambang tracking dan pembuatan ID ByteTrack |
| `person_core.py` | Default ukuran input YOLO 416, target deteksi 10 FPS, batas analisis 6 orang |
| `action_config.py` | Lokasi classifier dan **ambang aplikasi 70%** |
| `mixed_model_config.py` | Pemeriksaan paket ekspor dan letterbox hitam sesuai training |
| `models/.../best_model/inference_config.json` | Profil ekspor: 16 frame/2 detik, stride 0,5 detik, threshold validation 58% |

Threshold aplikasi adalah override yang terpisah dari threshold ekspor. Jangan
mengedit paket ekspor untuk mengubah ambang aplikasi: hash-nya diperiksa loader.
Kelas `0 = person` sudah diperiksa pada `YoloDetector.pt` yang dipakai saat integrasi.

## API yang tersedia

| Endpoint | Fungsi |
|---|---|
| `GET /api/health` | Memeriksa apakah backend aktif |
| `GET /api/config` | Metadata detector dan classifier; termasuk threshold aplikasi/ekspor |
| `POST /api/sessions/start` | Memulai analisis file atau webcam |
| `POST /api/sessions/{id}/stop` | Meminta sesi berhenti |
| `GET /api/sessions/{id}/stream` | Preview MJPEG |
| `WS /api/sessions/{id}/events` | Status sesi, statistik dan alarm |
| `GET /api/sessions` | Daftar riwayat sesi tersimpan |
| `GET /api/sessions/{id}` | Detail sesi aktif atau riwayat |
| `POST /api/sessions/{id}/review` | Menyimpan tinjauan alarm |
| `GET /api/sessions/{id}/files/{filename}` | Mengambil berkas hasil yang diizinkan |
| `GET /api/sessions/{id}/download` | Mengunduh ZIP hasil |

Form awal menerima `file` atau `camera_index`, `save_video`, `realtime`, dan
`analysis_scope`. Nilai scope adalah `scene` (default) atau `person`. Replay
file default offline; webcam selalu live. ID sesi aktif dan nama folder hasil
berbeda; gunakan field `output` untuk membuka riwayat/berkas hasil setelah selesai.

Satu sesi aktif diizinkan per proses; sesi tambahan mendapat HTTP 409. State
sesi/model berada dalam memori. Jalankan satu worker Uvicorn untuk versi ini.

## Output dan pengujian

Hasil disimpan pada `outputs/<sesi>/`: CSV prediksi/alarm, JSON ringkasan, preview,
video anotasi bila diminta, dan tinjauan. Tidak semua jenis keluaran tersedia
pada setiap mode; webcam tidak direkam oleh opsi penyimpanan video file.

```powershell
.\.venv-yolo\Scripts\python.exe -m pytest -q
.\.venv-yolo\Scripts\python.exe tests\smoke_real_models.py "D:\video\contoh.mp4"
```

Tes otomatis memakai model palsu untuk sebagian besar kontrak API/engine. Script
smoke memakai bobot asli dan membutuhkan video lokal serta paket model lengkap.
Ia menghasilkan `outputs/integration_real_smoke.json`; ini pemeriksaan fungsi,
bukan evaluasi akurasi. Rincian pengujian terdahulu berada di
[INTEGRASI_MODEL.md](INTEGRASI_MODEL.md).

## Masalah yang sudah diketahui

- Preview offline berhenti sejenak saat menunggu klip dianalisis di CPU.
- Ukuran YOLO aplikasi 416 berbeda dari training 640; evaluasi sebelum mengubah default.
- Pergantian ID bisa menghasilkan beberapa alarm untuk kejadian yang sama.
- Track hilang memakai area terakhir yang ditandai khusus, bukan posisi orang terkonfirmasi.
- Error timestamp webcam pernah dilaporkan; belum ada perubahan kode untuk masalah ini.
- Video hasil hanya bisa diputar jika memang disimpan dan codec didukung browser.
- Kode masih memakai satu origin; pemisahan hosting memerlukan URL API/WS dan CORS yang sesuai.

`README_ENGINE.md`, `VERIFIKASI.md` dan `MULAI.cmd` merujuk engine/entry point
desktop sebelumnya. Gunakan `run_dashboard.cmd` untuk aplikasi web gabungan.
