# ComfyUI-LongVideoChain

Membuat video panjang dari model yang hanya bisa ±15–20 detik per run: frame terakhir run N
otomatis menjadi frame pertama run N+1, run berikutnya diantrekan otomatis, dan di akhir semua
segmen disambung menjadi satu video (dengan audio).

## Instalasi
1. Salin folder `ComfyUI-LongVideoChain` ke `ComfyUI/custom_nodes/` (atau install lewat Git URL di ComfyUI Manager).
2. Restart ComfyUI. Tidak ada dependensi tambahan (memakai `av`, `Pillow`, `numpy` yang sudah dibawa ComfyUI).
3. Buka `workflows/MiniMax_H3_LongVideoChain.json` (workflow yang sudah disambungkan).

## Node
| Node | Fungsi |
|---|---|
| **LVC Segment Start** | Segmen 1 → gambar asli Anda. Segmen 2+ → frame terakhir segmen sebelumnya (PNG lossless). Juga memilih prompt untuk segmen ini. |
| **LVC Prompt Segment** | Satu kotak prompt untuk **satu** segmen. Rantai beberapa node ini, sambungkan yang terakhir ke `prompt_list` di Start. |
| **LVC Segment Save** | Menerima hasil **VAE Decode** (frame yang sudah selesai dirender) + audio. Menyimpan segmen, menyimpan frame benih, mengantre run berikutnya, dan di segmen terakhir menyambung semuanya. |
| **LVC Stitch** | Menyambung segmen yang sudah ada sekarang juga (selesai lebih awal / re-stitch). |

## Cara pakai
1. Isi `total_segments` di **Start** (mis. 5 segmen × ±20 dtk ≈ 100 dtk). Durasi/`length` di workflow = durasi **per segmen**.
2. Isi prompt per segmen, pilih salah satu cara:
   * **Kotak terpisah (disarankan untuk prompt panjang):** satu node **LVC Prompt Segment** per segmen. Node 1 tidak punya `previous`; node 2 menyambung `previous` ← node 1, dst.; node terakhir → `prompt_list` di Start. Segmen 1 memakai kotak 1, segmen 2 kotak 2, dst. Kotak kosong memakai prompt sebelumnya; jika kotak kurang dari segmen, kotak terakhir dipakai ulang. Untuk menambah segmen: salin node (Ctrl+C lalu Ctrl+Shift+V) dan sambungkan.
   * **Satu teks:** isi `prompts` di Start, dipisah baris berisi `---` (diabaikan bila `prompt_list` tersambung).
   Model hanya menerima prompt segmen yang sedang berjalan; prompt segmen lain tidak pernah digabung.
3. Biarkan `mode = new_chain`, lalu tekan **Run satu kali** (batch count = 1). Sisanya otomatis.
4. Hasil: `output/long_video_chain/<chain_id>/final_<waktu>.mp4`

## `mode` di node Start
| mode | Arti |
|---|---|
| **new_chain** (default) | Setiap kali Anda menekan Run, video **baru** dimulai dari gambar awal, di folder baru `long_video_chain/<chain_id>__<tanggal_jam>`. Sisa file dari percobaan lama (walau `chain_id` dan gambarnya sama) **tidak pernah ikut terpakai**. |
| **continue** | Melanjutkan rantai terbaru yang belum selesai. Run otomatis berikutnya diantrekan dengan mode ini oleh node Save (Anda tidak perlu mengubah apa pun). Pakai manual hanya untuk **melanjutkan setelah error/berhenti**. Jika rantai terbaru sudah selesai, otomatis dimulai rantai baru. |

Node Save terkunci ke folder rantai yang diberikan Start, jadi folder lain (run kedua, file hasil sinkronisasi) tidak bisa
mengacaukan rantai yang sedang berjalan.

## Menghentikan / melanjutkan
* **Berhenti**: Cancel run yang sedang jalan + kosongkan antrean. Atau buat file kosong bernama `STOP` di folder rantai.
* **Lanjut** setelah error/berhenti: ubah `mode` ke `continue`, lalu Run. Segmen yang sudah selesai tidak diulang. Setelah selesai, kembalikan ke `new_chain` untuk video berikutnya.
* **Mulai ulang**: `mode = new_chain` lalu Run. Folder lama tidak disentuh atau dihapus (hapus manual bila tidak perlu).

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

## Folder kerja terpisah & pemulihan (berguna di Colab)
Secara default semua file kerja disimpan di `output/long_video_chain/<chain_id>__<waktu>/`. Set variabel lingkungan
`LVC_WORK_DIR` ke folder lain jika folder output Anda disinkronkan ke tempat lain; file kerja disimpan di sana
dan **hanya video akhir** ditaruh di folder output.

Semua file kerja ditulis **sekali dan tidak pernah diubah** (`state_0000.json`, `state_0001.json`, ... satu per segmen;
`done.json` saat selesai). Jadi aman disalin/disinkronkan ke penyimpanan yang tidak menimpa file (mis. Drive yang
menyimpan "nama (2).ext" bila isi berubah). Jika folder kerja dipulihkan di mesin baru, `mode = continue` langsung
lanjut dari segmen berikutnya dengan frame terakhir yang sama persis. Jika ada file yang belum selesai terunduh, node
berhenti dengan pesan jelas (bukan melanjutkan dengan data yang salah).

## Batasan yang perlu diketahui
* Kualitas bisa pelan-pelan bergeser (warna/detail) setelah banyak segmen, karena tiap segmen dimulai dari satu gambar. Gambar referensi (wajah dsb.) di workflow sengaja tetap memakai gambar **asli**, bukan frame hasil, supaya identitas tidak ikut melenceng.
* Audio dibuat terpisah per segmen oleh model, jadi suara/musik bisa "melompat" di sambungan.
* Antrean otomatis memakai endpoint `/prompt` bawaan ComfyUI dari sisi server (browser tidak perlu terbuka). Seed di-random ulang tiap segmen (`randomize_seeds`).
* Jangan menekan Run lagi selagi rantai masih berjalan: itu memulai video **kedua** yang antreannya berselang-seling dengan yang pertama (keduanya selesai dengan benar, tetapi GPU dipakai dua kali lipat). Setiap run lanjutan terkunci ke folder rantainya sendiri, jadi dua rantai tidak saling mengacaukan.
* Run lanjutan diantrekan dari workflow yang sama; node Save hanya mengirim `class_type`/`inputs`/`_meta` (data internal ComfyUI seperti `is_changed` sengaja dibuang, kalau tidak ComfyUI memakai ulang hasil cache node Start dan segmen yang sama terulang).
