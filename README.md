# Prototipe deteksi dugaan orang jatuh — beberapa orang dalam satu kamera

Proyek deteksi orang jatuh di koridor atau ruang tunggu rumah sakit. Versi ini
mendeteksi beberapa orang, menggambar **17 titik tubuh per orang**, memberi ID
sementara melalui **ByteTrack**, dan mengekspor riwayat titik tubuh per ID.
Model bawaan adalah **YOLO11n-Pose**; tidak ada training pose pada tahap ini.
Aturan gerak per ID memberi alarm lokal ketika seseorang tampak turun cepat,
berakhir horizontal dekat pijakan awal, dan tetap demikian selama 1,5 detik.

**Alarm berarti dugaan jatuh, bukan kepastian medis.** Petugas perlu memeriksa
rekaman/orang tersebut. Pose yang hilang saja tidak dianggap jatuh.

## Menjalankan

Klik dua kali **MULAI.cmd**. Pilih **Pilih & proses video** atau **Mulai webcam**.
Ubah **Batas orang dalam gambar** jika diperlukan (awal: 10). Batas ini membatasi
jumlah deteksi, bukan jaminan semua orang terlihat atau terlacak. Tekan **Hentikan**
untuk menyelesaikan sesi dan **Buka hasil** untuk melihat ekspor.

Jika aturan terpenuhi, kotak ID berubah merah, banner alarm merah muncul, dan
komputer berbunyi lalu mengulang setiap 4 detik sampai **Tandai ditangani**
ditekan. Tombol itu menghentikan bunyi di aplikasi; kejadian tetap dicatat.
Alarm ini hanya tampil/berbunyi di komputer yang menjalankan aplikasi. Belum
ada pengiriman pesan ke ponsel, nurse station, atau sistem rumah sakit.

Pemrosesan berjalan lokal pada CPU. Webcam baru aktif setelah tombol ditekan.
Python dan paket tersimpan di `.venv-yolo`; lingkungan MediaPipe `.venv` tetap
tersedia. Paket dan model diunduh pada penggunaan pertama; setelah itu dapat
dipakai tanpa koneksi internet. Python 3.12 dengan Tcl/Tk disarankan.

Contoh pemrosesan tanpa tampilan:

```powershell
.\.venv-yolo\Scripts\python.exe extract_pose.py "D:\video\contoh.mp4"
.\.venv-yolo\Scripts\python.exe extract_pose.py "D:\video\contoh.mp4" --max-people 15 --no-video
.\.venv-yolo\Scripts\python.exe extract_pose.py "D:\video\contoh.mp4" --image-size 960 --max-frames 60
```

Ukuran gambar lebih tinggi dapat membantu orang yang tampak kecil, tetapi harus
diuji terhadap kecepatan. `--device 0` hanya berlaku jika pengguna memasang
PyTorch dengan dukungan CUDA; pemasangan bawaan proyek ini menggunakan CPU.

## Keluaran

Setiap sesi memiliki folder baru `outputs/<waktu>_yolo_<id>/`:

- `annotated.mp4`: video berkerangka dan ID, jika diaktifkan. Audio tidak disalin.
- `landmarks.csv`: semua orang, dengan kolom `track_id` dan `segment_id`.
- `tracks/track_0001.csv`, dst.: riwayat titik tubuh masing-masing ID.
- `frames.csv`: satu baris per frame, termasuk saat tidak ada orang terdeteksi.
- `detections.csv`: kotak semua deteksi, termasuk yang belum mendapat ID.
- `alerts.csv`: waktu, ID, dan bukti numerik setiap dugaan jatuh yang terkonfirmasi.
- `summary.json`: versi model, jumlah frame, kecepatan, dan statistik tiap ID.
- `preview.jpg`: frame terakhir.

Webcam mengekspor CSV, ringkasan, dan preview; video webcam tidak direkam.
Semua frame video yang berhasil dibaca diproses. Preview boleh melewati tampilan
frame agar antarmuka tetap responsif, tetapi ekspor tidak melewati frame tersebut.

## Bagaimana keputusan jatuh dibuat

Aturan dalam `fall_rules.py` memakai waktu video/kamera, sehingga kecepatan
pemrosesan komputer tidak mengubah durasi pengamatan. Untuk satu ID, sistem:

1. Menemukan pose tubuh awal yang cukup jelas dan tegak.
2. Dalam 0,25–1,5 detik, bahu atau hidung harus turun setidaknya 27% dari tinggi
   kotak tubuh awal, dengan laju minimal 32% tinggi tubuh per detik.
3. Kotak tubuh harus melebar/mendatar, dan tepi bawahnya tetap dekat dengan
   pijakan awal (tidak naik melebihi 10% tinggi tubuh awal). Ini membantu
   memisahkan jatuh ke lantai dari rebah ke ranjang atau sofa yang lebih tinggi.
4. Posisi mendatar itu harus berlangsung minimal 1,5 detik, dengan setidaknya
   0,8 detik bukti kotak terdeteksi. Celah pelacakan maksimal 0,6 detik ditoleransi.

Jika ByteTrack kehilangan ID saat tubuh sudah rebah, aturan boleh melanjutkan
konfirmasi dengan deteksi tanpa ID yang kotaknya saling tumpang tindih minimal
25% dan muncul dalam 0,6 detik di lokasi yang sama. Tanda `*` di video berarti
ID untuk alarm tersebut sedang disambungkan sementara. Data pose tanpa ID
tetap tidak diberi ID buatan di CSV titik tubuh. `alerts.csv` mencatat apakah
konfirmasi alarm memakai kotak sementara tersebut (`evidence_source`).

Satu kejadian hanya mengeluarkan satu alarm untuk ID yang sama. Sistem dapat
siap lagi setelah orang tampak tegak selama 1,5 detik. Nilai ambang adalah
konfigurasi awal untuk prototipe, bukan hasil kalibrasi CCTV rumah sakit.
Jika pose awal, gerak turun, atau kotak akhir tidak cukup jelas, sistem bisa
tidak memberi alarm. Gerakan sengaja yang cepat hingga berbaring di lantai
juga dapat memenuhi aturan; petugas tetap harus memeriksa. ID yang tertukar
karena orang berpapasan dapat memengaruhi hasil.

Aturan ini membutuhkan gerakan jatuh yang benar-benar terlihat dari kamera
tetap. Video yang baru mulai menyorot orang setelah ia tergeletak tidak
memiliki riwayat penurunan tubuh. Sudut kamera dari atas juga dapat membuat
orang rebah tampak tegak dalam koordinat gambar 2D. Kondisi itu memerlukan
detektor keadaan rebah dan kalibrasi kamera/area lantai terpisah, atau model
klasifikasi video yang diuji pada sudut serupa.

### Format data versi 2

Ada 17 baris per orang per frame. `x`, `y`, dan `bbox_*` dinormalisasi terhadap
lebar/tinggi gambar. `confidence` adalah skor titik tubuh. Tidak ada koordinat
3D; format ini berbeda dari ekspor 33 titik MediaPipe sebelumnya.

- `observed=1`: ID terlihat pada frame ini.
- `observed=0`, `track_status=temporarily_missing`: ID baru hilang (hingga 1 detik);
  koordinat kosong, tidak menggunakan posisi terakhir sebagai posisi saat ini.
- `keypoint_valid=1`: titik finite, di dalam gambar, dan confidence minimal 0.3.
- `pose_usable=1`: setidaknya 6 titik valid, termasuk 2 titik dari bahu/pinggul.
  Ini hanya aturan kualitas data, bukan keputusan jatuh.
- `segment_id`: berubah setelah kehilangan pose atau pose berkualitas rendah.
  Jangan membuat rangkaian training melintasi batas segmen atau sesi.

Jika tidak ada deteksi, `frames.csv` tetap mencatat frame. Deteksi yang belum
mendapat ID ditampilkan sebagai **Menunggu ID** dan dihitung di `unassigned_people`;
tidak diberi identitas buatan di CSV titik tubuh. Kotaknya tetap dicatat di
`detections.csv` agar keputusan alarm dapat diuji ulang tanpa menjalankan YOLO.

ID hanya berlaku di sesi dan kamera yang sama. Satu orang dapat mendapat ID baru
setelah tertutup atau keluar gambar; `unique_track_ids` bukan jumlah pasien unik.
ByteTrack juga dapat menukar ID ketika orang berpapasan. `segment_id` memisahkan
celah yang diketahui, tetapi tidak menjamin setiap pertukaran identitas terdeteksi.
Ambang awal ID baru dipasang pada skor deteksi 0.6 untuk mengurangi ID palsu;
orang yang kecil atau tertutup mungkin belum mendapat ID sampai terlihat lebih jelas.
ID yang sudah terbentuk tetap dapat dipulihkan lewat deteksi berskor lebih rendah.

## Pengujian dan langkah berikutnya

```powershell
.\.venv-yolo\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-yolo\Scripts\python.exe tests\check_video_pipeline.py data\gmdcsa_fall_01.mp4 data\gmdcsa_adl_02.mp4
```

Uji integrasi memerlukan `data/ultralytics_bus.jpg` dari sumber di bawah. Klip
buatan dari gambar ini hanya memeriksa beberapa orang, ID, celah data, dan ekspor;
bukan bukti akurasi pada CCTV rumah sakit. Webcam fisik belum diuji.

Berikutnya: rekam simulasi 3–5 orang bergerak/berpapasan pada sudut kamera yang
dituju, beri label waktu jatuh dan bukan-jatuh (termasuk duduk cepat, membungkuk,
berbaring di bangku/lantai), lalu ukur sensitivitas, alarm palsu per jam, dan
waktu respons. Setel ambang dengan data latih; nilai akhir harus diperiksa pada
video/pemeran lain. Classifier gerak terlatih dapat dibandingkan sebagai tahap
lanjutan. Integrasi pemberitahuan ke petugas memerlukan kanal komunikasi nyata.

Kode versi awal MediaPipe disimpan sebagai cadangan di `legacy_mediapipe_*`.
Hasil sesi lama tidak diubah; perhatikan `schema_version` sebelum menggabungkan CSV.

## Sumber dan atribusi

- YOLO11 Pose: https://docs.ultralytics.com/models/yolo11/
- Pelacakan ByteTrack: https://docs.ultralytics.com/modes/track/
- Ultralytics dan model YOLO11 menggunakan AGPL-3.0 atau lisensi Enterprise:
  https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Model resmi: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
- Gambar pengujian: https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg
- Video GMDCSA-24: https://zenodo.org/records/13354453

File video/gambar sampel dan bobot model diabaikan Git. Cantumkan sumber dataset
dalam laporan dan periksa izin sebelum mendistribusikan ulang datanya.
