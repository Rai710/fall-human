# Fall Detection — Dashboard + engine per-orang

Gabungan dua proyek:

- **UI**: dashboard React (`frontend/`) + API FastAPI (`backend/`) dari `fall-human-feat-dashboard`.
- **Engine**: `person_engine.py`, `person_core.py`, `live_core.py`, `pose_engine.py` dari `fall-human-BON`
  (YOLO → ByteTrack → klip 16 frame per orang → VideoMAE → alarm).

```text
Browser (React) ──HTTP/WS──> backend/ (FastAPI) ──> person_engine.process_people
                                                      ├─ Model 1: YOLO deteksi orang  (model_config.json)
                                                      └─ Model 2: VideoMAE deteksi jatuh (Hugging Face)
```

## Menjalankan (Windows)

1. `setup_dashboard.cmd` (sekali saja: membuat `.venv-yolo`, memasang dependensi).
2. `run_dashboard.cmd` → buka http://localhost:8000.
3. Pilih webcam atau upload video → **Mulai monitoring**.

Sesi pertama mengunduh dua hal otomatis: `yolo11n-pose.pt` (±6 MB) dan model VideoMAE dari Hugging Face
(±345 MB, `yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect`). Setelah itu jalan offline.
Aplikasi desktop lama tetap ada: `MULAI.cmd`.

Mode pengembangan UI: `uvicorn backend.main:app --port 8000` + `npm --prefix frontend run dev` (http://localhost:5173).

## Ganti model deteksi ORANG (Model 1)

Semua diatur di **`model_config.json`**. Default: `yolo11n-pose` bawaan Ultralytics.

| `source` | Artinya | Yang perlu diisi |
| --- | --- | --- |
| `pretrained` (default) | `yolo11n-pose.pt`, diunduh otomatis ke `models/` | — |
| `local` | `best.pt` hasil train sendiri | taruh di `models/best.pt` atau ubah `local.path` |
| `huggingface` | `best.pt` di repo Hugging Face | `huggingface.repo_id` (+ `filename`, `revision`) |

Contoh memakai model sendiri dari file lokal:

```json
{ "source": "local", "local": { "path": "models/best.pt" }, "person_class_ids": [0] }
```

Contoh dari Hugging Face (repo privat: set env `HF_TOKEN` dulu):

```json
{ "source": "huggingface",
  "huggingface": { "repo_id": "usernamemu/human-detector", "filename": "best.pt" },
  "person_class_ids": [0] }
```

Tanpa edit file, bisa lewat environment variable, mis. `set DETECTOR_SOURCE=local` lalu jalankan
`run_dashboard.cmd`. Variabel lain: `DETECTOR_LOCAL_PATH`, `DETECTOR_HF_REPO`, `DETECTOR_HF_FILENAME`,
`DETECTOR_HF_REVISION`, `DETECTOR_CLASSES`, `HF_TOKEN`.
`python download_detector.py` menyiapkan model tanpa membuka aplikasi.

**Perhatikan `person_class_ids`.** Ini nomor kelas "orang" di modelmu (lihat `names:` di `data.yaml` saat train).
`yolo11n-pose` memakai `0`. Jika modelmu punya kelas lain atau urutannya beda, isi sesuai. Salah nomor akan
ditolak dengan pesan yang menyebut daftar kelas model. Model deteksi biasa (bukan pose) tidak masalah:
engine hanya memakai kotak, bukan keypoint.

Model deteksi **jatuh** (Model 2) tetap VideoMAE dari Hugging Face seperti di zip BON. Saat model temanmu
siap, titik gantinya ada di `pose_engine.py` (`MODEL_NAME` / `LOCAL_MODEL`) atau folder lewat `model_path`.

## Yang diubah dari dua zip asal

- `person_engine.py`: detektor tidak lagi hard-coded `best_v5.pt` dan tidak dipaksa `task="pose"`;
  kelas orang & confidence dari config; event `session` (folder keluaran) dan `active_alarms` untuk UI.
- `detector_config.py`, `model_config.json`, `download_detector.py`: baru.
- `backend/runner.py`: sekarang memanggil `process_people` (sebelumnya `process_source`) dan memetakan
  statistik/alarm ke format UI; mengonversi `annotated.mp4` ke H.264 agar bisa diputar di browser
  (butuh `ffmpeg` atau `imageio-ffmpeg`, sudah ada di `requirements.txt`).
- `backend/repository.py`: menyelaraskan nama kolom CSV engine (`clip_end_ms` → `timestamp_ms`, dst.).
- `frontend/`: tampilan sama; ditambah ID orang di log alert/riwayat, kartu "Orang terlacak", info detektor
  aktif, dan centang "Ikuti waktu video (simulasi live)" untuk upload.

## Tes

```powershell
.\.venv-yolo\Scripts\python.exe -m pytest -q
```

Tes integrasi memakai YOLO/VideoMAE palsu; tidak mengunduh model.

## Catatan penting

- **Belum diuji dengan model sungguhan.** Rangkaian API → engine → riwayat diuji dengan model palsu,
  dan frontend lolos build/lint. Jalankan satu video nyata sebelum percaya hasilnya.
- Di CPU, VideoMAE ±1–2 detik per klip per orang. Dengan "simulasi live" aktif, hasil yang lebih tua dari
  3 detik dibuang dan alarm bisa terlewat; matikan centang itu untuk analisis lengkap (lebih lambat dari waktu video).
- "api HF" di sini berarti **mengunduh bobot dari Hugging Face Hub**, lalu dijalankan lokal. Memanggil model
  yang di-host di Inference Endpoint/Space (tanpa unduh) belum disediakan.
- Lisensi: Ultralytics AGPL-3.0, checkpoint VideoMAE CC-BY-NC-4.0 (non-komersial). Detail di `README_ENGINE.md`.
