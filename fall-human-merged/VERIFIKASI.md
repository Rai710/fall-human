# Verifikasi integrasi per orang — 20 Oktober 2026

## Apa yang dikerjakan

Backend lokal YOLO11 Pose + ByteTrack dihubungkan ke VideoMAE per ID/segmen.
Perubahan teman dan remote Git belum diambil/digabungkan. Pengujian memakai
model yang sudah tersedia di laptop, tanpa unduhan, API, webcam, atau RTSP.
Runtime: torch 2.14.1+cpu, torchvision 0.29.1+cpu, ultralytics 8.4.182,
lap 0.5.13, transformers 5.20.1. CUDA tidak tersedia pada runtime ini.

## Hasil pengujian akhir pada klip lokal

Konfigurasi: YOLO 416, target deteksi 10 FPS, maksimum 6 orang, 16 sampel/2 detik,
interval per orang 1 detik, ambang FallDown 0,75, konfirmasi 1 hasil,
batas usia hasil live 3 detik. Tidak ada training model.

| Klip / mode | Orang puncak | Detektor+tracker rata-rata | VideoMAE rata-rata / klip orang | Hasil |
| --- | ---: | ---: | ---: | --- |
| Kosong / offline | 0 | 42 ms | tidak dijalankan | 0 alarm |
| Jatuh / offline | 1 | 34 ms | 1,583 detik | 1 alarm; skor tertinggi FallDown 96,55% |
| Aktivitas biasa / offline | 1 | 33 ms | 1,547 detik | 0 alarm; kelas LyingDown |
| Fixture beberapa orang / offline | 3 | 33 ms | 1,557 detik | 3 klip dianalisis; 0 alarm |
| Jatuh / simulasi live | 1 | 44 ms | 1,599 detik | 1 alarm; 3 klip dianalisis |
| Fixture beberapa orang / simulasi live | 3 | 56 ms | 1,633 detik | 2 klip dihitung; 1 hasil terlambat, 1 klip kedaluwarsa |

Pada jatuh live, klip pemicu membentang 2,176–4,160 detik sumber. Prediksi klip
tersebut memerlukan 1,594 detik; hasil diterima 2,765 detik setelah frame akhir
klip. Ini belum mengukur waktu sejak awal jatuh; awal jatuh belum diberi label
manual. Konfirmasi dua hasil belum diuji pada konfigurasi akhir ini.

Jeda terpanjang antar callback preview pada dua uji live akhir sekitar 47 ms
(jatuh) dan 63 ms (fixture). Ini mengukur callback engine, bukan FPS aktual
render GUI, kualitas jaringan, atau jaminan kelancaran CCTV.
Dua replay akhir tidak mengalami keterlambatan pembaca lebih dari 1 detik.

Semua frame file yang berhasil dibaca tersimpan pada annotated.mp4 sesuai
jumlah frame sumber yang diuji. Overlay memakai hasil yang sudah tersedia
saat penulisan, sehingga prediksi yang baru selesai setelah EOF hanya muncul
pada preview akhir/log/banner GUI, bukan dipasang mundur pada video.

## Kesimpulan teknis

- Detector/tracker lokal cukup ringan pada contoh ini; VideoMAE tetap menjadi
  hambatan utama. Target analisis setiap orang setiap 1 detik belum tercapai.
- Satu orang sudah dapat memicu dugaan jatuh dalam simulasi live sampel ini.
- Pada 3 orang, kapasitas CPU kurang: antrean mengakibatkan hasil terlambat atau
  klip kedaluwarsa. Preview lancar tidak sama dengan penilaian semua orang tepat waktu.
- Sampel jatuh dan aktivitas biasa sangat sedikit. Fixture multi-orang berasal
  dari gambar contoh Ultralytics, bukan kejadian jatuh ramai di rumah sakit.
  Angka ini tidak boleh disebut akurasi atau validasi klinis.
- GUI, bunyi alarm, webcam, RTSP, putus jaringan, dan operasi berjam-jam belum
  diuji secara interaktif. Alarm di benchmark diterima melalui callback/log.

## Masalah yang ditemukan dan diperbaiki

Uji awal melewatkan jatuh di mode live karena klip lengkap yang masih menunggu
ikut dibuang ketika orang tidak terdeteksi. Kini hanya buffer parsial yang
terputus; klip lengkap tetap dinilai selama belum kedaluwarsa. Hasil memakai ID
historis dan tidak dipasang ke segmen baru. Celah pendek tidak diisi gambar palsu.

Ukuran detector diturunkan dari 640 ke 416 untuk uji CPU. Input ByteTrack menerima
skor rendah (0,1) dan ambang pembentukan ID 0,35; ini tidak membuat skor rendah
menjadi bukti jatuh. Keputusan jatuh tetap berasal dari VideoMAE.
Konfirmasi awal diatur 1 hasil sebagai dugaan cepat; ini menambah risiko alarm
palsu dibanding meminta dua hasil. Pilihan 2 tersedia di GUI/CLI.

Salah satu batch awal mengalami jeda pengukuran sangat besar dan tidak
representatif. Angka batch tersebut tidak dipakai pada tabel di atas; pengukuran
akhir diulang dengan timestamp dan keterlambatan pembaca yang tercatat.

## Bukti dan pengulangan

Hasil lengkap disalin ke `outputs/benchmark_people_20261020_1536/`, termasuk
`benchmark_report.json`, CSV, summary, preview, dan annotated video setiap sesi.
Model dan data uji tidak dimasukkan ke Git.

13 tes inti/integrasi mencakup pemisahan segmen, antrean bergiliran, klip
kedaluwarsa, hasil lama tidak menempel ke segmen baru, klip lengkap tetap dinilai
saat deteksi hilang, serta indikator alarm historis tanpa kotak orang saat ini.
Tes ini tidak mengukur akurasi model.

```powershell
.\.venv-yolo\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv-yolo\Scripts\python.exe tests\benchmark_people.py
```

Versi kode sebelum perubahan dicadangkan di `.cache/person_backup/`.
