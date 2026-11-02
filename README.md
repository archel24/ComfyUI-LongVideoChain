# ComfyUI-LongVideoChain

Membuat video panjang dari model yang hanya bisa ±15–20 detik per run: frame terakhir run N
otomatis menjadi frame pertama run N+1, run berikutnya diantrekan otomatis, dan di akhir semua
segmen disambung menjadi satu video (dengan audio).

## Instalasi
1. Salin folder `ComfyUI-LongVideoChain` ke `ComfyUI/custom_nodes/`
2. Restart ComfyUI. Tidak ada dependensi tambahan (memakai `av`, `Pillow`, `numpy` yang sudah dibawa ComfyUI).
3. Buka `workflows/MiniMax_H3_LongVideoChain.json` (workflow Anda yang sudah disambungkan).

## Node
| Node | Fungsi |
|---|---|
| **LVC Segment Start** | Run 1 → gambar asli Anda. Run 2+ → frame terakhir run sebelumnya (PNG lossless). Juga memilih prompt untuk segmen ini. |
| **LVC Segment Save** | Menerima hasil **VAE Decode** (frame yang sudah selesai dirender) + audio. Menyimpan segmen, menyimpan frame benih, mengantre run berikutnya, dan di segmen terakhir menyambung semuanya. |
| **LVC Stitch** | Menyambung segmen yang sudah ada sekarang juga (selesai lebih awal / re-stitch). |

## Cara pakai
1. Isi `total_segments` di **Start** (mis. 5 segmen × ±20 dtk ≈ 100 dtk).
2. Isi `prompts` di **Start**: satu prompt per segmen, dipisah baris berisi `---`. Jika prompt lebih sedikit dari segmen, prompt terakhir dipakai ulang.
3. Tekan **Run satu kali** (batch count = 1). Sisanya otomatis.
4. Hasil: `output/long_video_chain/<chain_id>/final_<waktu>.mp4`

## Menghentikan / melanjutkan
* **Berhenti**: Cancel run yang sedang jalan + kosongkan antrean. Atau buat file kosong bernama `STOP` di folder chain.
* **Lanjut** setelah error/berhenti: cukup tekan Run lagi dengan `chain_id` yang sama. Segmen yang sudah selesai tidak diulang.
* **Mulai ulang**: ganti `chain_id`, atau ganti gambar awal (rantai lama otomatis dipindah ke folder `..__archived_..`, tidak dihapus). Rantai yang sudah selesai juga otomatis diarsipkan saat Run lagi.

## Tentang "frame terakhir yang belum jadi"
Node Save mengambil frame dari **output VAE Decode** (setelah sampling selesai), bukan dari preview/latent
di tengah proses. Frame benih ditulis sebagai **PNG lossless langsung dari tensor** (bukan dari file mp4),
jadi yang masuk ke segmen berikutnya identik piksel-per-piksel dengan frame terakhir yang dirender.
Jika frame paling akhir dari model terlihat lembek/berkurang kualitasnya, naikkan `tail_trim` (1–3):
beberapa frame terakhir dibuang dan benih diambil dari frame yang lebih bersih (tanpa pengulangan frame).

## Detail penyambungan
* Frame pertama segmen 2+ (≈ gambar benih) dibuang agar tidak ada frame ganda di sambungan; audio dipotong sebanyak yang sama sehingga tetap sinkron.
* Semua segmen disambung dengan encode tunggal di akhir (`final_crf`, default 16). File antara memakai `segment_crf` (default 12).
* Jika satu segmen resolusinya berbeda, frame-nya diubah ukurannya mengikuti segmen pertama.

## Folder kerja terpisah (opsional, berguna di Colab)
Secara default semua file kerja (state, frame benih, video per segmen) disimpan di
`output/long_video_chain/<chain_id>/`. Jika folder output Anda disinkronkan ke tempat lain (mis. Google Drive),
set variabel lingkungan `LVC_WORK_DIR` ke folder lokal lain sebelum menjalankan ComfyUI. File kerja akan
disimpan di sana, dan **hanya video akhir** yang ditaruh di folder output.

## Batasan yang perlu diketahui
* Kualitas bisa pelan-pelan bergeser (warna/detail) setelah banyak segmen, karena tiap segmen dimulai dari satu gambar. Gambar referensi (wajah dsb.) di workflow sengaja tetap memakai gambar **asli**, bukan frame hasil, supaya identitas tidak ikut melenceng.
* Audio dibuat terpisah per segmen oleh model, jadi suara/musik bisa "melompat" di sambungan.
* Antrean otomatis memakai endpoint `/prompt` bawaan ComfyUI dari sisi server (browser tidak perlu terbuka). Seed di-random ulang tiap segmen (`randomize_seeds`).
