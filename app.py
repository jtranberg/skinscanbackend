import os
import json
import traceback
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import requests

# === App setup ===
app = Flask(__name__)
CORS(app)

# === Paths ===
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
TRIAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model3.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels_8.json')
TREATMENTS_PATH = os.path.join(MODEL_DIR, 'treatments.json')

# === GitHub Release URLs ===
GH_RELEASE = "https://github.com/jtranberg/skinscanbackend/releases/download/v1.0.0"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"⬇️ Downloading {url} → {dest}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Downloaded {dest}")

# === Download if missing ===
download_file(f"{GH_RELEASE}/best_model.keras", MODEL_PATH)
download_file(f"{GH_RELEASE}/best_model3.keras", TRIAGE_MODEL_PATH)
download_file(f"{GH_RELEASE}/class_labels_8.json", LABELS_PATH)
download_file(f"{GH_RELEASE}/treatments.json", TREATMENTS_PATH)

# === Load models and configs ===
model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
triage_model = load_model(TRIAGE_MODEL_PATH) if os.path.exists(TRIAGE_MODEL_PATH) else None
class_labels = json.load(open(LABELS_PATH)) if os.path.exists(LABELS_PATH) else ["Unknown"]
treatments = json.load(open(TREATMENTS_PATH)) if os.path.exists(TREATMENTS_PATH) else {}

if model:
    print(f"✅ Main model loaded from {MODEL_PATH}")
if triage_model:
    print(f"✅ Triage model loaded from {TRIAGE_MODEL_PATH}")

# === Predict route remains unchanged ===
# (leave your existing /predict route code as-is)

# === Launch ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Python model microservice running on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
