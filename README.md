# 📡 SentimenWatch — Analisis Sentimen Proposal Akses Udara Militer AS

> **Proyek Akademik Mahasiswa** — Bukan situs resmi pemerintah atau institusi apapun.

Website Flask untuk menganalisis sentimen masyarakat Indonesia terhadap isu Proposal Akses Udara Militer Amerika Serikat, menggunakan model **Bidirectional LSTM** yang dilatih pada 11.273 komentar publik dari YouTube & Instagram.

---

## 🗂️ Struktur Folder

```
Sentimen_Web_App/
├── app.py                    ← Flask Backend Utama
├── save_tokenizer.py         ← Script helper untuk save model dari Jupyter
├── requirements.txt          ← Dependencies (tensorflow-cpu untuk Vercel)
├── vercel.json               ← Konfigurasi deploy Vercel
├── .gitignore
├── model/
│   ├── lstm_model.h5         ← [WAJIB] Model LSTM hasil training
│   └── tokenizer.json        ← [WAJIB] Tokenizer tersimpan
├── static/
│   ├── css/
│   │   └── style.css         ← Styling tema Maroon & Putih
│   └── images/               ← Letakkan semua gambar visualisasi di sini
│       ├── viz1_distribusi_sentimen.png
│       ├── viz2_time_series.png
│       ├── viz3_wordcloud.png
│       ├── viz4_top_words.png
│       ├── viz5_platform.png
│       ├── confusion_matrix.png
│       └── learning_curves.png
├── templates/
│   ├── layout.html           ← Base template (header, navbar, footer)
│   └── index.html            ← Halaman utama (dashboard + prediksi)
└── data_raw_komentar.csv     ← [OPSIONAL] Data mentah untuk statistik
```

---

## ⚙️ Langkah Setup

### 1. Simpan Model dari Jupyter

Jalankan `save_tokenizer.py` di Jupyter setelah training selesai:

```python
# Di Jupyter Notebook (setelah training)
import json, os

os.makedirs('model', exist_ok=True)

# Simpan tokenizer
with open('model/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer.to_json())

# Simpan model
model.save('model/lstm_model.h5')

print("✅ Model & tokenizer tersimpan!")
```

### 2. Siapkan Gambar Visualisasi

Salin semua gambar PNG hasil visualisasi ke `static/images/`:

```bash
cp viz1_distribusi_sentimen.png static/images/
cp viz2_time_series.png         static/images/
cp viz3_wordcloud.png           static/images/
cp viz4_top_words.png           static/images/
cp viz5_platform.png            static/images/
cp confusion_matrix.png         static/images/
cp learning_curves.png          static/images/
```

### 3. Install Dependencies & Jalankan Lokal

```bash
# Buat virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Jalankan Flask
python app.py
# Buka http://localhost:5000
```

---

## 🚀 Deploy ke Vercel

### Persiapan
1. Push project ke GitHub (pastikan `model/*.h5` dan `model/tokenizer.json` ikut ter-push)
2. Untuk file model besar (>100MB), gunakan **Git LFS**:
   ```bash
   git lfs install
   git lfs track "model/*.h5"
   git add .gitattributes
   ```

### Deploy
1. Buka [vercel.com](https://vercel.com) → Login dengan GitHub
2. **New Project** → Import repository ini
3. Framework: **Other** (bukan Next.js)
4. **Deploy** — Vercel akan otomatis baca `vercel.json`

### ⚠️ Catatan Limit Vercel
- Vercel Hobby: **500MB** deployment size
- `tensorflow-cpu` (~200MB) + model H5 + dependencies ≈ sekitar 300-400MB
- Jika melebihi limit, pertimbangkan:
  - **Railway.app** atau **Render.com** (gratis, tanpa size limit ketat)
  - Konversi model ke **ONNX** (jauh lebih kecil)

---

## 🔌 API Endpoint

| Method | Endpoint   | Deskripsi                              |
|--------|------------|----------------------------------------|
| GET    | `/`        | Halaman utama (dashboard + prediksi)   |
| POST   | `/predict` | Prediksi sentimen teks (JSON)          |
| GET    | `/health`  | Status server & model                  |

### Contoh Request `/predict`

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Ini pengkhianatan! Udara Indonesia bukan milik asing!"}'
```

### Contoh Response

```json
{
  "label": "Negatif",
  "confidence": 94.2,
  "detail": {
    "Negatif": 94.2,
    "Netral": 3.8,
    "Positif": 2.0
  },
  "cleaned_text": "khianat udar indonesia milik asing"
}
```

---

## 🧠 Arsitektur Model

- **Embedding Layer**: vocab_size=15000, dim=128
- **SpatialDropout1D**: 0.3
- **Bidirectional LSTM**: 128 units, dropout=0.3
- **Dense Output**: 3 kelas (Positif / Negatif / Netral) — Softmax
- **Data**: 11.273 komentar, rasio 85.3% Negatif / 7.4% Positif / 7.3% Netral

---

*© 2025 — Proyek Akademik Mahasiswa · Tidak berafiliasi dengan pemerintah manapun*
