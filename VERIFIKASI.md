# Verifikasi YOLO Pose + ByteTrack + aturan dugaan jatuh

Pengujian lokal, 19 Oktober 2026. Python 3.12.14, PyTorch 2.14.1 CPU,
Ultralytics 8.4.182, YOLO11n-Pose, masukan model 640 piksel.
Versi paket lengkap tersedia di `requirements.lock.txt`.

## Hasil uji fungsional

| Sumber | Frame | Maksimal orang dengan ID bersamaan | Catatan |
| --- | ---: | ---: | --- |
| Klip contoh beberapa orang | 46 | 3 | ID 1, 2, 3 konsisten pada 24 frame awal |
| GMDCSA-24 Subject 1 Fall 01 | 203 | 1 | Setidaknya satu pose cukup jelas pada 174 frame |
| GMDCSA-24 Subject 1 ADL 02 | 198 | 1 | Setidaknya satu pose cukup jelas pada 198 frame |

Kecepatan rata-rata pemanggilan inferensi/pelacakan sekitar 8.5 FPS pada klip
beberapa orang, 21.3 FPS pada Fall 01, dan 22.4 FPS pada ADL 02. Angka ini bukan
jaminan kinerja kamera langsung; pemanggilan pertama juga mencakup pemanasan model.
Kecepatan seluruh pipeline dicatat terpisah sebagai `processing_fps` di ringkasan sesi.

Klip beberapa orang dibuat dari gambar contoh Ultralytics yang digeser sedikit,
diselingi 10 frame kosong, lalu ditampilkan kembali. Ini menguji sambungan
deteksi, pelacakan, serta ekspor; bukan evaluasi orang berpapasan atau CCTV rumah sakit.
Orang yang sebagian besar berada di luar gambar mungkin belum mendapat ID.

Uji awal menemukan ID palsu di sekitar tubuh dan kursi pada video Fall 01.
Ambang pembentukan ID baru diperketat dari 0.25 menjadi 0.6. Uji ulang pada video
yang sama menghasilkan maksimal satu ID bersamaan. Sampel ini dipakai untuk
penyetelan dan pemeriksaan fungsi, sehingga bukan data evaluasi independen.

## Pemeriksaan yang lolos

- Enam uji data: orang berurutan berbeda, pose hilang, ID lama kembali, kualitas
  titik rendah, deteksi tanpa ID, serta normalisasi koordinat.
- 17 baris per orang per frame; setiap CSV individual sama dengan data ID tersebut
  di CSV gabungan. Riwayat antar-ID tidak dicampur.
- `frames.csv` tetap mencatat frame tanpa deteksi; timestamp meningkat dan sesuai CSV titik tubuh.
- Jumlah frame video ekspor sama dengan jumlah frame yang diproses.
- Video kosong tidak menghasilkan ID atau koordinat tubuh buatan.
- Penghentian setelah 3 frame menyelesaikan CSV dan ringkasan dengan status `stopped`.
- Inisialisasi antarmuka Tkinter berhasil.

## Penambahan aturan jatuh

Uji ulang dari video mentah melalui YOLO, ByteTrack, aturan, ekspor, dan callback
antarmuka menghasilkan:

| Video | Frame | Alarm dugaan jatuh | Waktu alarm |
| --- | ---: | ---: | ---: |
| GMDCSA-24 Subject 1 Fall 01 | 203 | 1 | 5,424 detik, frame 160 |
| GMDCSA-24 Subject 1 ADL 02 (berbaring ke ranjang) | 198 | 0 | — |

Pada alarm Fall 01, penurunan bahu/hidung yang dipakai setara 44,9% tinggi
kotak tubuh awal dalam 1,376 detik. Postur horizontal bertahan 1,504 detik
sebelum alarm. `alerts.csv`, `summary.json`, callback alarm, dan video beranotasi
memberi jumlah kejadian yang sama. Frame terakhir video jatuh diperiksa secara
visual: banner dan kotak ID berwarna merah tampak jelas.

Sebanyak 14 uji fungsi lolos (6 riwayat ID dan 8 aturan jatuh), termasuk
penolakan berbaring di tempat lebih tinggi, pose hilang setelah mendarat,
durasi horizontal terlalu singkat, orang kedua yang tetap tegak, dan tidak
adanya gerak turun sebelumnya. Dua uji baru memeriksa sambungan kotak tanpa ID
di lokasi yang sama dan penolakan kotak tanpa ID di lokasi yang jauh.
Inisialisasi GUI serta tombol tanda ditangani
berhasil. Perangkat audio fisik dan webcam langsung belum diuji.

## Uji pada video yang dipilih pengguna

| Video | Hasil aturan lama | Hasil setelah perbaikan | Penjelasan |
| --- | ---: | ---: | --- |
| `02.mp4` | 0 alarm | 1 alarm, frame 116/detik 3,920 | Kandidat jatuh terlihat sekitar detik 2,42; ID terputus saat di lantai. Kotak tanpa ID di lokasi yang sama melanjutkan konfirmasi. |
| `Orang-Jatuh-terpeleset-SnapYT.App.mp4` | 0 alarm | 0 alarm | Kamera berpindah ke orang yang sudah di tanah. Momen jatuh tidak terlihat; sudut dari atas membuat pose 2D tampak tegak. |

Uji ulang pada GMDCSA Fall 01 tetap menghasilkan satu alarm pada frame 160,
sementara GMDCSA ADL 02 tetap nol. Frame akhir `02.mp4` diperiksa secara visual:
banner merah dan kotak merah bertanda `ID 1*` terlihat pada orang yang rebah.
Tanda bintang berarti pencocokan kotak sementara, bukan ID ByteTrack yang pulih.
`detections.csv` yang baru menyimpan kotak tanpa ID. Pemutaran ulang aturan dari
CSV saja menghasilkan frame alarm yang sama (116 pada `02.mp4`) dan tetap nol
pada klip ADL, sehingga keputusan dapat diaudit tanpa menjalankan YOLO lagi.
Klip YouTube tidak dihitung sebagai negatif sejati untuk evaluasi akurasi;
status orang tergeletak memang ada tetapi kejadian jatuhnya tidak terekam.

Angka 1/1 pada dua video ini **bukan** ukuran sensitivitas atau tingkat alarm
palsu. Video contoh yang sama dipakai untuk memilih aturan awal dan pemeriksaan
fungsi, sehingga belum ada evaluasi independen. Video koridor rumah sakit,
variasi sudut kamera, berpapasan, dan aktivitas yang mirip jatuh perlu diuji
sebelum digunakan untuk pemantauan nyata.

Laporan angka yang dapat dibaca program: `outputs/yolo_verification.json`.
Hasil MediaPipe sebelumnya tetap berada di folder sesi lama.

## Batas yang belum diuji

Webcam fisik, CCTV rumah sakit, orang yang berpapasan, dan tertutup dalam waktu
lama belum diuji. ID dapat berganti atau tertukar. Proporsi pose yang cukup jelas
bukan akurasi jatuh. Alarm saat ini lokal pada komputer aplikasi; tidak ada
pengiriman pesan otomatis ke petugas rumah sakit.

Sumber gambar pengujian:
https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg

Sumber video:
https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos
