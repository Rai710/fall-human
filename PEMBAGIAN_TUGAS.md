# Pembagian tugas tim

Tim terdiri dari **Raihan Buono**, **Michael**, dan **Rioo**. Tabel berikut
menetapkan tanggung jawab utama tiap anggota; integrasi backend/API, pengujian
sistem, serta penyusunan laporan dikerjakan bersama.

## Penanggung jawab utama

| Anggota | Fokus | Hasil yang diserahkan |
|---|---|---|
| **Raihan Buono — model jatuh** | Dataset/label Le2i + CAUCAFall, fine-tuning VideoMAE, preprocessing, threshold, evaluasi jatuh | Paket ekspor lengkap, pemetaan label, panduan input, laporan metrik dan contoh kesalahan |
| **Michael — deteksi dan tracking** | Dataset YOLO, `YoloDetector.pt`, pengujian orang kecil/tertutup, ByteTrack dan kestabilan ID | Bobot + daftar kelas, konfigurasi detector/tracker, perbandingan input 416/640, laporan orang terlewat dan ID berganti |
| **Rioo — frontend** | React, tampilan upload, progres, preview, alarm, riwayat, review dan unduh hasil | Antarmuka dashboard, integrasi panggilan API dari frontend, dan skenario uji antarmuka |

PIC berarti penjaga perubahan modul, bukan satu-satunya orang yang mengerjakannya.
Raihan Buono dan Michael menyediakan serta menguji fungsi inferensi modul
masing-masing. Rioo mengerjakan antarmuka dan pemanggilan API dari frontend.
Penyambungan fungsi ke backend/API serta masalah integrasi ditangani bersama.

## Kepemilikan modul untuk mengurangi konflik

| Area | PIC | Reviewer |
|---|---|---|
| `pose_engine.py`, `mixed_model_config.py`, `action_config.py` | Raihan Buono | Rioo untuk kontrak; Michael untuk input per orang |
| `detector_config.py`, `person_engine.py`, `person_core.py`, `tracker.yaml` | Michael | Raihan Buono untuk crop dan jendela waktu |
| `frontend/` | Rioo | Raihan Buono dan Michael untuk makna hasil yang ditampilkan |
| `backend/`, `live_core.py`, dependensi dan format output | Bersama; pilih satu PIC per perubahan | Minimal satu anggota lain |
| Skenario demo, evaluasi gabungan, README dan laporan akhir | Semua anggota | Saling memeriksa hasil dan klaim |

Satu orang menjadi PIC integrasi untuk setiap rilis/demo. Tugasnya menggabungkan
perubahan dan mencatat hasil uji, bukan mengambil semua pekerjaan backend.

## Kontrak yang perlu disepakati

1. **Detector:** kelas orang, format kotak, confidence, ukuran input, dan perangkat yang dipakai.
2. **Tracking:** ID sementara, kapan track dianggap hilang, dan cara menandai bukti area terakhir.
3. **VideoMAE:** 16 frame RGB/sekitar 2 detik, letterbox hitam 224×224, processor, label `no_fall`/`fall`.
4. **Alarm:** skor `fall` minimal 70%, satu konfirmasi saat ini, aturan pemulihan dan duplikasi ID.
5. **API:** `analysis_scope`, status sesi, error, skor, timestamp, ID/area, dan lokasi keluaran.

Menaikkan confidence YOLO tidak sama dengan menaikkan threshold jatuh. Model
yang diganti harus membawa konfigurasi label dan preprocessing yang cocok.

## Prioritas untuk laporan dan demo saat ini

| Prioritas | Pekerjaan | PIC | Selesai bila |
|---|---|---|---|
| 1 | Tetapkan satu versi source dan paket model untuk demo | Bersama | Versi/commit, bobot, threshold, mode, dan perangkat dicatat |
| 2 | Evaluasi jatuh vs aktivitas normal pada video yang sama | Raihan Buono | Prediksi/alarm benar dan salah dicatat; bukan hanya memilih contoh berhasil |
| 3 | Bandingkan YOLO input 416 dan 640 pada contoh yang sulit | Michael | Orang terlewat, kotak salah, ID berganti, dan waktu proses dibandingkan |
| 4 | Pastikan upload → proses → riwayat → unduh berjalan | Rioo, dibantu tim untuk backend | Skenario berhasil/gagal dapat diulang tanpa memakai model palsu saja |
| 5 | Susun hasil dan batas sistem | Semua | Tiap anggota menulis bagian modulnya; istilah dan angka konsisten |

Untuk tahap laporan, pertahankan arsitektur yang sudah ada. Optimasi GPU,
retraining besar, dan deployment cloud dapat dijadikan pekerjaan berikutnya
setelah kebutuhan demo dan hasil pengujian tercatat jelas.

## Cara bekerja di Git

- Sepakati `final` sebagai branch integrasi dan `main` sebagai versi yang siap dibagikan/demo.
- Kerjakan perubahan pada branch fitur masing-masing dari versi dasar yang disepakati.
- Buat perubahan kecil per modul; beri tahu PIC sebelum menyentuh file bersama.
- Review dan uji perubahan sebelum digabung; gunakan merge/PR agar riwayat tim terjaga.
- Simpan source, konfigurasi, tes dan dokumentasi di Git. Bagikan bobot/dataset terpisah.
- Catat asal/revisi/hash bobot agar ketiga anggota menguji model yang sama.

## Bukti untuk laporan

Setiap percobaan sebaiknya mencatat video/skenario, mode analisis, versi bobot,
threshold, jumlah orang, perangkat, waktu proses, alarm dan kesalahan. Evaluasi
detector membahas orang yang terlewat/kotak salah; evaluasi classifier membahas
jatuh vs nonjatuh; evaluasi integrasi membahas ketepatan ID, alarm, dan alur web.

Metrik per klip, jumlah alarm, dan jumlah kejadian adalah ukuran berbeda.
Hasil crop per orang juga dilaporkan terpisah dari hasil adegan penuh.
