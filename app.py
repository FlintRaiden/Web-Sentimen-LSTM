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

# ── Path absolut (penting untuk Vercel/Linux) ─────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, 'model', 'lstm_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'model', 'tokenizer.json')
CSV_PATH       = os.path.join(BASE_DIR, 'data_labeled.csv')

# ── Lazy-load TF & Keras (hemat memory saat startup) ─────────
tf        = None
model     = None
tokenizer = None

def load_model_and_tokenizer():
    """Load TF model & tokenizer sekali saja (singleton)."""
    global tf, model, tokenizer
    if model is not None:
        return True
    try:
        # Paksa Keras 2 agar kompatibel dengan file .h5 lama
        os.environ['TF_USE_LEGACY_KERAS'] = '1'

        import tensorflow as _tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        tf = _tf

        if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
            print(f"[ERROR] File tidak ditemukan:\n  Model   : {MODEL_PATH}\n  Tokenizer: {TOKENIZER_PATH}")
            return False

        model = load_model(MODEL_PATH)

        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer = tokenizer_from_json(f.read())

        print(f"[OK] Model & tokenizer berhasil dimuat.")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return False

# ── Preprocessing Indonesia ───────────────────────────────────
import nltk
from nltk.corpus import stopwords

# Unduh resource NLTK ke folder yang bisa ditulis di Vercel
NLTK_DATA_DIR = os.path.join(BASE_DIR, 'nltk_data')
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_DIR)
nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)
nltk.download('punkt',     download_dir=NLTK_DATA_DIR, quiet=True)

STOPWORDS_ID = set(stopwords.words('indonesian'))
CUSTOM_STOPWORDS = {
    'yg','yang','aja','sih','deh','nih','kah','lah',
    'gue','gw','lu','lo','kamu','aku','dia','mereka',
    'ini','itu','juga','udah','sudah','sudahkah','belum',
    'mau','bisa','ada','tidak','tak','ga','gak','nggak',
    'tapi','kalau','kalo','sama','terus','lagi',
    'ya','yah','oh','ah','wah','kan','dong','emang',
    'dgn','dg','utk','utuk','buat','dari','dan','atau',
    'ke','di','untuk','dengan','yang','pada','adalah',
    're','rt','rp','via','amp'
}
ALL_STOPWORDS = STOPWORDS_ID | CUSTOM_STOPWORDS

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
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 1]
    stemmer = get_stemmer()
    if stemmer:
        tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ── Konfigurasi LSTM ──────────────────────────────────────────
MAX_LEN   = 100
LABEL_MAP = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

def predict_sentiment(text: str):
    if not load_model_and_tokenizer():
        return None, None, None
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    cleaned = preprocess_text(text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    probs   = model.predict(padded, verbose=0)[0]
    idx     = int(np.argmax(probs))
    label   = LABEL_MAP.get(idx, 'Tidak Diketahui')
    detail  = {
        'Negatif': round(float(probs[0]) * 100, 2),
        'Netral' : round(float(probs[1]) * 100, 2),
        'Positif': round(float(probs[2]) * 100, 2),
    }
    return label, round(float(probs[idx]) * 100, 2), detail

# ── Statistik CSV ─────────────────────────────────────────────
def get_dashboard_stats():
    try:
        if not os.path.exists(CSV_PATH):
            print(f"[DEBUG] CSV tidak ditemukan: {CSV_PATH}")
            return None
        df = pd.read_csv(CSV_PATH)
        if 'sentimen' not in df.columns:
            print("[DEBUG] Kolom 'sentimen' tidak ada")
            return None
        label_counts = df['sentimen'].value_counts().to_dict()
        print(f"[DEBUG] Sentimen: {label_counts}")
        return {
            'total'         : len(df),
            'label_counts'  : label_counts,
            'platform_stats': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {}
        }
    except Exception as e:
        print(f"[ERROR] Gagal baca CSV: {e}")
        return None

# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def index():
    stats       = get_dashboard_stats()
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)
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
        return jsonify({'error': 'Model belum tersedia.'}), 503
    return jsonify({
        'label'       : label,
        'confidence'  : confidence,
        'detail'      : detail,
        'cleaned_text': preprocess_text(text),
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None,
                    'model_path': MODEL_PATH, 'model_exists': os.path.exists(MODEL_PATH)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)