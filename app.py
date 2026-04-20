# ============================================================
# app.py — Flask Backend: Analisis Sentimen LSTM
# Isu Proposal Akses Udara Militer Amerika di Indonesia
# ============================================================

import os
import re
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── Lazy-load TF & Keras (hemat memory saat startup) ────────
tf = None
model = None
tokenizer = None

def load_model_and_tokenizer():
    """Load TF model & tokenizer sekali saja (singleton)."""
    global tf, model, tokenizer
    if model is not None:
        return True
    try:
        import tensorflow as _tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        tf = _tf

        MODEL_PATH     = os.path.join('model', 'lstm_model.h5')
        TOKENIZER_PATH = os.path.join('model', 'tokenizer.json')

        if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
            return False

        model = load_model(MODEL_PATH)

        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        return True
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return False

# ── Preprocessing Indonesia ──────────────────────────────────
import nltk
from nltk.corpus import stopwords

# Unduh resource NLTK jika belum ada
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

STOPWORDS_ID = set(stopwords.words('indonesian'))
CUSTOM_STOPWORDS = {
    'yg','yang','aja','sih','deh','nih','kah','lah',
    'gue','gw','lu','lo','kamu','aku','dia','mereka',
    'ini','itu','juga','udah','sudah','sudahkah','belum',
    'mau','bisa','ada','tidak','tak','ga','gak','nggak',
    'tapi','tapi','kalau','kalo','sama','terus','lagi',
    'ya','yah','oh','ah','wah','kan','dong','emang',
    'dgn','dg','utk','utuk','buat','dari','dan','atau',
    'ke','di','untuk','dengan','yang','pada','adalah',
    're','rt','rp','via','amp'
}
ALL_STOPWORDS = STOPWORDS_ID | CUSTOM_STOPWORDS

# Sastrawi stemmer (load sekali)
_stemmer = None

def get_stemmer():
    global _stemmer
    if _stemmer is None:
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            _stemmer = StemmerFactory().create_stemmer()
        except ImportError:
            _stemmer = None
    return _stemmer

def preprocess_text(text: str) -> str:
    """Preprocessing: case fold → clean → tokenize → stopword → stem."""
    # 1. Case folding
    text = text.lower()
    # 2. Hapus URL, mention, hashtag
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    # 3. Hapus karakter non-alfabet
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 4. Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Tokenisasi sederhana
    tokens = text.split()
    # 6. Buang stopwords
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 1]
    # 7. Stemming (Sastrawi)
    stemmer = get_stemmer()
    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ── Konfigurasi LSTM (harus sama dengan saat training) ───────
MAX_LEN    = 100
LABEL_MAP  = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

def predict_sentiment(text: str):
    """Prediksi sentimen teks menggunakan model LSTM."""
    if not load_model_and_tokenizer():
        return None, None, None

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = preprocess_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

    probs   = model.predict(padded, verbose=0)[0]
    idx     = int(np.argmax(probs))
    label   = LABEL_MAP.get(idx, 'Tidak Diketahui')
    confidence = float(probs[idx]) * 100

    detail = {
        'Negatif': round(float(probs[0]) * 100, 2),
        'Netral' : round(float(probs[1]) * 100, 2),
        'Positif': round(float(probs[2]) * 100, 2),
    }
    return label, round(confidence, 2), detail

# ── Baca statistik CSV untuk Dashboard ───────────────────────
def get_dashboard_stats():
    try:
        # Gunakan path absolut agar tidak salah lokasi
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, 'data_labeled.csv') 

        if not os.path.exists(csv_path):
            print(f"[DEBUG] File tidak ditemukan di: {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        
        # Pastikan kolom 'sentimen' ada
        if 'sentimen' not in df.columns:
            print("[DEBUG] Kolom 'sentimen' tidak ditemukan di CSV")
            return None

        total = len(df)
        # Menghitung jumlah label dan konversi ke dictionary
        label_counts = df['sentimen'].value_counts().to_dict()
        
        # DEBUG: Print ke terminal untuk cek isi data
        print(f"[DEBUG] Data Sentimen Ditemukan: {label_counts}")

        return {
            'total': total,
            'label_counts': label_counts,
            # Tambahkan default 0 jika platform tidak ada
            'platform_stats': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {}
        }
    except Exception as e:
        print(f"[ERROR] Gagal baca CSV: {e}")
        return None

# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def index():
    stats = get_dashboard_stats()
    model_ready = os.path.exists(os.path.join('model', 'lstm_model.h5')) and \
                  os.path.exists(os.path.join('model', 'tokenizer.json'))
    return render_template('index.html', stats=stats, model_ready=model_ready)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong.'}), 400

    if len(text) > 1000:
        return jsonify({'error': 'Teks terlalu panjang (maks 1000 karakter).'}), 400

    label, confidence, detail = predict_sentiment(text)

    if label is None:
        return jsonify({'error': 'Model belum tersedia. Pastikan file model sudah di-upload.'}), 503

    cleaned_text = preprocess_text(text)

    return jsonify({
        'label'       : label,
        'confidence'  : confidence,
        'detail'      : detail,
        'cleaned_text': cleaned_text,
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
