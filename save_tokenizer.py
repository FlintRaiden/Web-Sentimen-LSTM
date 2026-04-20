"""
save_tokenizer.py
─────────────────
Jalankan script ini di Jupyter Notebook SETELAH training selesai,
untuk menyimpan tokenizer ke format JSON yang dibutuhkan Flask app.

Cara penggunaan di Jupyter:
    %run save_tokenizer.py
    # atau copy-paste isinya ke cell baru
"""

import json
import os

# ── Pastikan variabel 'tokenizer' dan 'model' sudah ada di memory ──
# (hasil dari training LSTM)

# 1. Simpan Tokenizer ke JSON
tokenizer_json = tokenizer.to_json()
os.makedirs('model', exist_ok=True)

with open('model/tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)

print("✅ Tokenizer berhasil disimpan: model/tokenizer.json")

# 2. Simpan Model ke H5
model.save('model/lstm_model.h5')
print("✅ Model berhasil disimpan: model/lstm_model.h5")

# 3. Verifikasi — coba load kembali
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model

with open('model/tokenizer.json', 'r') as f:
    tok_loaded = tokenizer_from_json(f.read())

model_loaded = load_model('model/lstm_model.h5')

print(f"\n📊 Verifikasi Tokenizer:")
print(f"   Jumlah kata dalam vocab : {len(tok_loaded.word_index)}")
print(f"   Contoh mapping          : {dict(list(tok_loaded.word_index.items())[:5])}")

print(f"\n🧠 Verifikasi Model:")
model_loaded.summary()

print("\n✅ Semua file siap! Salin ke folder model/ di project Flask.")
print("   Struktur yang dibutuhkan:")
print("   Sentimen_Web_App/")
print("   └── model/")
print("       ├── lstm_model.h5")
print("       └── tokenizer.json")
