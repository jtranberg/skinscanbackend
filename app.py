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

# === App setup ===
app = Flask(__name__)
CORS(app)

# === Paths ===
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
TRIAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model3.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels_8.json')
TREATMENTS_PATH = os.path.join(MODEL_DIR, 'treatments.json')

# === Load Models & Configs ===
model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
triage_model = load_model(TRIAGE_MODEL_PATH) if os.path.exists(TRIAGE_MODEL_PATH) else None
class_labels = json.load(open(LABELS_PATH)) if os.path.exists(LABELS_PATH) else ["Unknown"]
treatments = json.load(open(TREATMENTS_PATH)) if os.path.exists(TREATMENTS_PATH) else {}

# === Predict Route ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        img = Image.open(request.files['image'].stream).convert('RGB')
        img = img.resize((256, 256))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        age = request.form.get('age')
        gender = request.form.get('gender')
        weight = request.form.get('weight')
        lat = request.form.get('lat')
        lon = request.form.get('lon')

        triage_class = "Unknown"
        if triage_model:
            triage_result = triage_model.predict(img_array)[0]
            triage_index = np.argmax(triage_result)
            triage_class = ["Normal", "Benign", "Malignant"][triage_index]

        predicted_class = "Unknown"
        confidence = 0.0
        top1 = top2 = top3 = "Unknown"

        if triage_class != "Normal" and model:
            prediction_probs = model.predict(img_array)[0]
            top_indices = prediction_probs.argsort()[-3:][::-1]
            top_labels = [class_labels[i] for i in top_indices]
            top_probs = [float(prediction_probs[i]) for i in top_indices]
            predicted_class = top_labels[0]
            confidence = top_probs[0]
            top1, top2, top3 = top_labels
        elif triage_class == "Normal":
            predicted_class = top1 = top2 = top3 = "Normal"
            confidence = 1.0

        treatment = treatments.get(predicted_class, "No treatment info available.")

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 4),
            'top1': top1,
            'top2': top2,
            'top3': top3,
            'age': age,
            'gender': gender,
            'weight': weight,
            'most_common_treatment': treatment,
            'suggested_clinics': [],  # This can be handled by a separate microservice if needed
            'suggested_doctors': []
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# === Launch ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\U0001F680 Python model microservice running on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
