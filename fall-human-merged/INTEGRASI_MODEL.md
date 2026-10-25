# Integrasi YOLO tim detector dan VideoMAE Le2i + CAUCAFall

Tanggal: 25 Oktober 2026 (Asia/Jakarta).

## Penyesuaian ambang aplikasi

Atas permintaan pengguna, ambang alarm aplikasi dinaikkan dari 58% menjadi
**70%** di `action_config.py`. Threshold hasil validation pada paket ekspor tetap
0,58; bobot, hash ekspor, sampling 16 frame/2 detik, stride 0,5 detik, dan satu
konfirmasi dipertahankan. API menampilkan `threshold=0.70` dan
`export_threshold=0.58`. Angka uji model asli di bagian bawah berasal dari
integrasi awal pada ambang 58%, bukan pengukuran ulang akurasi pada ambang 70%.
Tes kontrak API/runner diperbarui untuk memastikan kedua mode memakai 70%.

Pemeriksaan detector menemukan metadata training `imgsz=640`, sedangkan ukuran
input aplikasi masih 416. Uji perbandingan 416 versus 640 pada rekaman tujuan
disarankan sebelum mengubah default; ukuran lebih besar menambah beban CPU dan
tidak otomatis mengatasi perbedaan kamera, occlusion, atau pose di luar training.

## Tujuan dan lokasi

Menggabungkan dashboard React/FastAPI pada branch `final`, YOLO custom tim
detector, dan checkpoint VideoMAE dua kelas terbaru. Folder hasil:

`D:\Codinganku\web-fall-human\fall-human\fall-human-merged`

Repo Git berada satu tingkat di atas folder tersebut. Source versi final telah
dikirim ke `main` pada `Rai710/fall-human`, merge commit `f5c00d3`, dengan
riwayat main lama dipertahankan. Deployment cloud belum dilakukan.

## Model yang benar-benar dipakai

| Komponen | Model / sumber | Penggunaan |
|---|---|---|
| Detector | `models/YoloDetector.pt`, sudah berada di proyek web | Ultralytics task `detect`; kelas 0 `person`; 9.428.179 parameter |
| Tracker | `tracker.yaml`, ByteTrack | Menjaga ID sementara; tidak memiliki bobot classifier terpisah |
| Classifier | `models/best_model-20261025T045257Z-1-001/best_model` | VideoMAE epoch 4, `mixed_fine_tuning`, dua kelas `no_fall`/`fall` |

Paket classifier disalin lengkap dari:

`D:\Codinganku\DETEKSI ORANG JATUH\models\best_model-20261025T045257Z-1-001\best_model`

SHA-256 `model.safetensors`:
`7f1e01fb54deebf29dde027785c0d2fc6051f956ba85fdbf3c9af16466f51154`.
Loader memeriksa hash seluruh berkas yang terdaftar dalam `export_complete.json`.
Sumber model dan checkpoint asli tidak diubah atau dilatih ulang.

## Rencana dan keputusan implementasi

1. Pertahankan frontend, FastAPI, riwayat, codec H.264, dan pemilihan detector dari
   versi web. Jangan menimpa seluruh proyek dengan versi desktop.
2. Bawa dukungan classifier 2/3/7 kelas, profil input model biner, gate alarm
   biner, dan perbaikan track hilang dari proyek model.
3. Satukan pengaturan classifier di `action_config.py`; API dan runner membaca
   threshold/durasi/stride dari ekspor yang sama, bukan angka UI terpisah.
4. Sediakan dua mode. Seluruh kamera menjadi default karena sesuai training;
   per orang menjalankan kedua model dan ByteTrack, dengan status eksperimen.
5. Uji mock untuk kontrak API/scheduling, lalu video lokal dengan model asli;
   salin hasil teruji ke folder tujuan dan siapkan build React serta launcher.

## Perubahan sebelum dan sesudah

| Bagian | Sebelumnya pada versi web | Setelah integrasi |
|---|---|---|
| Path YOLO | `models/best.pt` yang tidak ada | `models/YoloDetector.pt` yang sudah dimuat/diperiksa |
| Task YOLO | Adapter web sudah mendukung detector biasa | Dukungan tersebut dipertahankan setelah membawa engine terbaru |
| Classifier dashboard | Default HF 7 kelas, label wajib `FallDown`, download otomatis | Paket lokal Le2i + CAUCAFall, label `no_fall`/`fall`, tanpa fallback download |
| Profil input | Profil model HF lama | 16 RGB / 2 detik; PIL bilinear letterbox hitam 224×224; processor ekspor |
| Threshold / interval | 0,75 / 1 detik pada default engine | 0,58 / 0,5 detik, dibaca dari `inference_config.json` |
| Gate alarm | Bergantung label `FallDown` | Mendukung biner berdasarkan skor fall dan threshold; dua hasil di bawah ambang memulihkan gate |
| Area analisis | Selalu crop per orang | Pilihan seluruh kamera / per orang; default seluruh kamera |
| Upload file | Simulasi live aktif | Upload dan offline menjadi pilihan awal; webcam tetap live |
| Track hilang | Belum memiliki penanganan area terakhir terbaru | ROI terakhir terbatas, bukti historis dibedakan dari deteksi orang saat ini |
| Informasi dashboard | Detector saja | Nama classifier, threshold, mode analisis; tracking tidak aktif pada mode scene |
| Alarm area | ID ditampilkan sebagai orang | Area terakhir ditandai sebagai area pada live/riwayat |
| Stop sesi | Active ID bisa dilepas meski worker belum berhenti | Sesi baru ditolak selama worker sebelumnya masih berjalan |
| Runtime lokal | Wajib `.venv-yolo` dalam proyek web | Mendukung `local_python.txt` untuk runtime terpisah yang sudah diuji |

## Alur dua mode

```text
React → FastAPI → SessionRunner
                   ├─ scene: adegan penuh → preprocessing ekspor → VideoMAE → alarm kamera
                   └─ person: YOLO → ByteTrack → klip/crop per ID → VideoMAE → alarm per ID/area
```

`scene` tidak menjalankan detector, sehingga tidak menentukan siapa yang jatuh.
`person` memakai YOLO dan classifier sekaligus, tetapi crop mengubah distribusi
input training. Hasil mode crop belum menjadi bukti akurasi multi-orang.

## File utama yang berubah

- `model_config.json`: path detector.
- `action_config.py`, `mixed_model_config.py`: profil classifier dan pemeriksaan ekspor.
- `pose_engine.py`, `live_core.py`: dukungan model biner, input, alarm, event web.
- `person_engine.py`, `person_core.py`: gabungan engine terbaru dan adapter detector/web.
- `backend/main.py`, `backend/runner.py`: mode API, pemilihan engine, model lokal, metadata alarm.
- `frontend/src/api.ts`, `Live.tsx`, `History.tsx`: kontrak dan kontrol mode/model, label area.
- `run_dashboard.ps1`: pemilihan interpreter lokal.
- `tests/test_web_mixed_model.py`: kontrak dua mode dan semantik biner.
- `tests/test_person_pipeline.py`, `test_lost_region.py`: regresi tracking terbaru.
- `tests/test_dashboard_integration.py`: uji pipeline per orang secara eksplisit.
- `tests/smoke_real_models.py`: uji model asli melalui upload API.

## Verifikasi aktual

**45 tes lolos**, satu peringatan deprecation dari TestClient Starlette/httpx.
Tes mencakup API, konfigurasi detector, kontrak kedua mode, binary threshold,
preprocessing, pemulihan gate, track hilang, serta pencegahan sesi tumpang tindih.
Laporan JUnit: `outputs/integration_checks/pytest.xml`.

**React/TypeScript/Vite lolos build** dengan `npm run build`; hasilnya berada di
`frontend/dist`. Lint tidak dijalankan sebagai pemeriksaan terpisah.
Launcher pada folder tujuan berhasil menyalakan Uvicorn di port 8000. Pemeriksaan
browser menampilkan `YoloDetector.pt`, classifier dua kelas, threshold 58%, mode
seluruh kamera sebagai default, serta simulasi live file dalam keadaan mati.

Kedua model asli diuji melalui FastAPI TestClient menggunakan file lokal
`gmdcsa_fall_01.mp4` (sekitar 6,87 detik), tanpa webcam, RTSP, atau download model:

| Mode | Frame dibaca/disimpan | Klip dianalisis | Alarm | Inferensi rata-rata |
|---|---|---|---|---|
| Seluruh kamera | 203 / 203 | 9 | 1 | 2.392 ms/klip |
| Per orang | 203 / 203 | 9 | 2 | 1.564 ms/klip |

Video hasil tersedia melalui API file, riwayat/prediksi terbaca, dan konversi H.264
berhasil pada kedua sesi. Kedua sesi memakai `finetuned2` dan threshold 0,58.
Laporan mentah: `outputs/integration_real_smoke.json`; folder sesi tercatat di sana.

Waktu keseluruhan scene 46,23 detik, person 23,92 detik. Ini bukan perbandingan
kecepatan yang adil: startup scene memuat model, sedangkan person memakai cache
classifier yang telah dimuat. Angka tersebut juga bukan latensi deteksi real-time.

Pada mode person, puncaknya satu orang tetapi terbentuk dua ID (1 dan 2), masing-
masing memicu alarm. Dua alarm bukan bukti dua kejadian jatuh; pergantian ID dapat
menggandakan alarm. Sistem juga memperingatkan bukti area terakhir dan kapasitas
serial di bawah target. Hal ini tetap menjadi keterbatasan, bukan disembunyikan.

Tes otomatis dan uji model dilakukan pada salinan integrasi sebelum pemasangan.
Skrip pemasangan membandingkan SHA-256 berkas yang disalin agar source hasil
pengujian dan source tujuan identik. Tidak ada klaim pengujian akurasi baru.

## Cara menjalankan

Klik `run_dashboard.cmd`, lalu buka `http://localhost:8000`. Pilih **Upload video**
dan mulai dari mode **Seluruh kamera**. Pilih **Per orang** untuk mencoba pipeline
YOLO + ByteTrack + VideoMAE. Tidak perlu menjalankan aplikasi desktop terpisah.

Pada komputer ini, `local_python.txt` menunjuk environment web terpisah di
`D:\Codinganku\DETEKSI ORANG JATUH\.cache\web-integration-env\Scripts\python.exe`.
Environment tersebut memakai pustaka ML yang sudah ada lewat `.pth`; paket web
dipasang terpisah, tanpa meng-upgrade environment training `.venv-yolo`.
Untuk komputer lain, buat environment mandiri lewat `setup_dashboard.cmd`.
File `local_python.txt` dan bobot tidak ikut Git.

Backup berkas tujuan sebelum pemasangan disimpan dalam `.cache/integration_backup_*/`.
Tidak ada penghapusan source/model lama. Source integrasi telah dibagikan ke
`main`; bobot, hasil, build, dan konfigurasi interpreter tetap diabaikan Git.

## Batas yang masih ada

- Model dapat salah menganggap rebah di kasur sebagai jatuh, seperti uji lokal terdahulu.
- Satu video uji tidak memvalidasi akurasi, multi-orang, alarm per jam, atau latensi live.
- Antrean internal membatasi kerja; satu sesi aktif per proses. Belum ada antrean multi-user persisten.
- Kamera index adalah kamera backend; belum ada streaming kamera browser ke server.
- Belum ada reconnect RTSP otomatis, deployment cloud, autentikasi publik, atau storage cloud.
- Domain API/WebSocket dan CORS perlu disesuaikan jika frontend/backend dipisah saat deploy.
