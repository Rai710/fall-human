# Deteksi dugaan orang jatuh — YOLO + ByteTrack + VideoMAE lokal

Backend utama mendeteksi orang dengan YOLO11 Pose, memberi ID sementara lewat
ByteTrack, lalu menilai klip RGB setiap orang menggunakan VideoMAE. MediaPipe dan
aturan sudut/kecepatan koordinat tubuh tidak dipakai untuk menentukan jatuh.

## Menjalankan

1. Klik `MULAI.cmd` di folder proyek.
2. Biarkan **Analisis per orang (YOLO + ByteTrack)** tercentang.
3. Biarkan **Simulasi live: ikuti waktu video** tercentang untuk mengukur perilaku
   yang mendekati sumber langsung. Mode offline menunggu analisis dan tidak
   boleh dijadikan bukti kecepatan live.
4. Pilih video CCTV. Setelah file berhasil, webcam atau RTSP bisa dicoba sendiri.
5. Perhatikan waktu deteksi/tracking, waktu analisis klip, usia hasil, dan antrean.

Model lokal yang dibutuhkan:

- `models/yolo11n-pose.pt`
- `models/videomae-fall/config.json`
- `models/videomae-fall/preprocessor_config.json`
- `models/videomae-fall/model.safetensors`

Model-model ini sudah ada saat integrasi dibuat. YOLO tidak diunduh otomatis.
Kotak **Izinkan unduhan model pertama** hanya mengizinkan unduhan VideoMAE.
Untuk menyiapkan VideoMAE secara terpisah, jalankan `download_model.py`.
Membuka GUI tidak mengaktifkan kamera. Dependensi yang belum ada dapat dipasang
launcher; environment lokal tetap bernama `.venv-yolo`.

## Alur dan pengaturan awal

```text
Pembaca kamera/file ──> preview terbaru dan perekaman file
                   └─> YOLO + ByteTrack ──> klip RGB per ID/segmen
                                           └─> VideoMAE ──> log + alarm
```

Pembacaan sumber, tracking, dan VideoMAE berjalan pada pekerja terpisah.
Setiap klip berisi 16 sampel nyata sekitar 2 detik. Ini tidak membatasi kamera
menjadi 16 FPS. Satu area crop tetap, berupa gabungan kotak orang sepanjang
klip dengan margin 20%, dipakai untuk semua sampel agar gerakan spasial tidak
hilang akibat gambar terus dipusatkan ulang. Crop dibuat persegi dengan padding.
Checkpoint yang semula dicoba pada seluruh kamera belum terbukti akurat pada
semua crop per orang; evaluasi pada rekaman tujuan tetap diperlukan.

| Pengaturan | Awal | Keterangan |
| --- | --- | --- |
| Target deteksi | 10 FPS | Target sampling YOLO, bukan jaminan FPS aktual |
| Ukuran YOLO | 416 | Model pose lokal tetap dipakai; titik pose tidak menentukan jatuh |
| Batas analisis orang | 6 | Orang di luar batas ditandai; bukan dianggap aman |
| Klip | 16 frame / sekitar 2 detik | Sampel hilang tidak diduplikasi |
| Jeda analisis per orang | 1 detik | Pengajuan kerja, bukan jaminan model selesai dalam 1 detik |
| Ambang FallDown | 75% | Skor checkpoint, bukan probabilitas klinis terkalibrasi |
| Konfirmasi awal | 1 hasil | Dugaan awal cepat; dapat dinaikkan menjadi 2 di GUI/CLI |
| Batas usia hasil live | 3 detik | Hasil lebih tua tidak mengonfirmasi/memulihkan alarm |

Satu hasil sebagai dugaan awal mengurangi waktu menunggu aksi singkat, tetapi
berisiko lebih banyak alarm palsu. Mengatur konfirmasi 2 menambah penundaan dan
bisa melewatkan jatuh yang hanya muncul pada satu jendela. Tidak ada pengaturan
yang sudah divalidasi untuk penggunaan klinis.

VideoMAE memproses satu klip aktif, dengan paling banyak satu klip terbaru
menunggu per orang. Penggantian klip orang A mempertahankan giliran orang B.
Klip yang sudah melewati batas usia sebelum mulai dihitung dicatat sebagai
`SKIPPED_EXPIRED`. Jika kapasitas CPU kurang, kejadian masih bisa terlewat;
antrean terbatas tidak membuat model menjadi lebih cepat.

## ID, kehilangan deteksi, dan alarm

- ID ByteTrack bersifat sementara, bukan identitas pasien.
- Celah pengamatan sampai 0,5 detik ditoleransi tanpa menggandakan gambar.
  Celah lebih panjang memulai segmen baru dan mengosongkan buffer parsial. Klip
  lengkap yang sudah terbentuk tetap boleh dinilai selama belum kedaluwarsa.
- Hasil klip nyata yang masih segar dapat memicu dugaan jatuh walaupun orang
  sudah tidak terdeteksi saat hasil selesai. ID/segmen kejadian tetap memakai
  ID historis; hasil tidak dipasang pada segmen/orang baru.
- Kotak tidak menjadi prediksi posisi masa kini. Jika tracking terlambat,
  tampilannya ditandai **TRACKING LAMA**.
- Satu kejadian dikunci sampai ada dua hasil normal berturut-turut. `LyingDown`
  sendiri tidak memicu alarm baru. Pergantian ID masih dapat memicu duplikasi.
- Banner dan bunyi pengingat tetap aktif sampai **Tandai ditangani** ditekan.
  Tombol ini tidak menghapus riwayat atau mengubah keputusan model.

Deteksi yang hilang saat tubuh berbaring, tubuh kecil/jauh, tertutup orang lain,
dan ID tertukar masih merupakan hambatan. Jumlah deteksi yang ditampilkan tidak
menjamin semua orang di ruangan terlihat. Batas komputasi juga bisa membuat
sebagian orang tidak dianalisis; periksa penanda pada layar dan CSV.

## Membaca waktu dengan benar

- **Deteksi/tracking ms:** satu panggilan YOLO + ByteTrack, termasuk transfer hasil.
- **Analisis ms:** processor + satu prediksi VideoMAE pada klip satu orang.
- **queue_ms:** waktu dari akhir klip sampai prediksi mulai, termasuk tracking/crop.
- **Usia hasil:** waktu sejak frame akhir klip diambil sampai hasil diterima.
- **Waktu alarm terhadap jatuh:** perlu label awal kejadian manual; belum sama
  dengan waktu analisis. Ada waktu pengumpulan klip dan konfirmasi juga.

Jika prediksi rata-rata 1,5 detik dan ada 3 orang dengan target tiap 1 detik,
beban serial sekitar 4,5 detik kerja per detik sumber. Itu belum memenuhi target.
Preview yang lancar juga tidak membuktikan alarm tiba tepat waktu.
Ringkasan menandai bila perkiraan beban serial melebihi kapasitas target.

Model/detector disiapkan sebelum sumber mulai dibaca. Waktu startup dipisah dari
waktu prediksi; pengukuran prediksi tetap mencakup panggilan pertama VideoMAE.
Replay dengan keterlambatan pembaca lebih dari 1 detik ditandai tidak stabil
(`replay_timing_valid: false`). Jangan memakai sesi tersebut sebagai benchmark
live yang representatif. CSV tetap menyimpan angka mentah.

## Keluaran

Setiap sesi berada di `outputs/<waktu>_people_<id>/`:

- `tracking.csv`: kotak piksel, ID/segmen, skor, waktu deteksi, usia tracking,
  serta `analysis_enabled` untuk orang dalam batas analisis.
- `predictions.csv`: ID/segmen, rentang klip, semua skor, waktu analisis/antrean,
  usia hasil, status identitas saat hasil tiba, dan klip yang kedaluwarsa.
- `alerts.csv`: ID kejadian, ID/segmen historis, kamera, rentang klip, waktu hasil.
- `summary.json`: konfigurasi, perangkat, rata-rata/p95, klip diganti/kedaluwarsa,
  kehilangan segmen, jumlah ID, dan peringatan kapasitas.
- `annotated.mp4`: semua frame file yang berhasil dibaca bila simpan aktif;
  overlay menunjukkan informasi yang tersedia saat frame ditulis, bukan label
  yang dipasang mundur setelah prediksi selesai. Audio tidak disalin.
- `preview.jpg`: tampilan terakhir.

Webcam/RTSP tidak direkam oleh opsi simpan file ini. URL CCTV tidak ditulis ke
ringkasan. Putus koneksi mengakhiri sesi; reconnect otomatis belum tersedia.
Model, data, output, environment, dan cache tetap diabaikan Git.

## CLI dan tes

```powershell
cd "D:\Codinganku\DETEKSI ORANG JATUH"
.\.venv-yolo\Scripts\python.exe extract_pose.py "D:\video\cctv.mp4" --no-video
.\.venv-yolo\Scripts\python.exe extract_pose.py "D:\video\cctv.mp4" --offline
.\.venv-yolo\Scripts\python.exe extract_pose.py --camera 0 --max-people 3
```

`--scene` memilih backend seluruh kamera sebagai pembanding. GUI menyediakan
pilihan yang sama dengan mematikan **Analisis per orang**. Opsi tambahan:
`--detection-fps`, `--max-people`, `--detector`, `--interval`, `--window`,
`--threshold`, `--confirm`, `--max-age`, `--model-dir`, dan `--allow-download`.
Menurunkan target FPS sampai sampel tidak cukup akan membuat klip tidak terbentuk.

```powershell
.\.venv-yolo\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv-yolo\Scripts\python.exe tests\benchmark_people.py
```

Benchmark memakai klip lokal yang tersedia, tanpa webcam atau unduhan. Klip
multi-orang adalah fixture integrasi dari gambar contoh, bukan rekaman jatuh
ramai di rumah sakit. Tes backend lama ada di `tests/legacy_pose/` sebagai arsip.
`requirements.lock.txt` masih snapshot environment sebelumnya; launcher memakai
`requirements.txt`. Hasil pengujian aktual ada di `VERIFIKASI.md`.

## Sumber

- VideoMAE fall checkpoint (CC-BY-NC-4.0):
  https://huggingface.co/yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect
- Ultralytics YOLO/tracking (AGPL-3.0; opsi lisensi komersial terpisah):
  https://docs.ultralytics.com/modes/track/
- ByteTrack: https://github.com/ifzhang/ByteTrack
- Sampel GMDCSA-24: https://zenodo.org/records/13354453

Dataset latihan checkpoint VideoMAE belum dijelaskan lengkap oleh pembuatnya.
Sistem ini prototipe, belum punya evaluasi akurasi pada CCTV rumah sakit.
