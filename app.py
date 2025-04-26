import os
import json
import traceback
import numpy as np
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

app = Flask(__name__)
CORS(app)

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)

MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
TRIAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model3.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels_8.json')
TREATMENTS_PATH = os.path.join(MODEL_DIR, 'treatments.json')

MODEL_URL = "https://github.com/jtranberg/8_class_model/releases/download/v1.0/best_model.keras"
TRIAGE_URL = "https://github.com/jtranberg/3_class_model/releases/download/v1.0/best_model3.keras"

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"⬇️ Downloading {dest} ...")
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded: {dest}")

download_file(MODEL_URL, MODEL_PATH)
download_file(TRIAGE_URL, TRIAGE_MODEL_PATH)

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
triage_model = load_model(TRIAGE_MODEL_PATH) if os.path.exists(TRIAGE_MODEL_PATH) else None
class_labels = json.load(open(LABELS_PATH)) if os.path.exists(LABELS_PATH) else ["Unknown"]
treatments = json.load(open(TREATMENTS_PATH)) if os.path.exists(TREATMENTS_PATH) else {}


print(f"📍 Received location: lat={lat}, lon={lon}")

# === Gemini Clinic + Doctor Suggestion ===
def get_clinics_and_doctors(lat, lon):
    prompt = f"""
You are a helpful medical assistant AI.

Based on the location (latitude: {lat}, longitude: {lon}), list the top 3 nearby dermatology clinics and doctors.

Respond ONLY in **valid JSON format**, like this:
{{
  "clinics": [
    {{
      "name": "ClearSkin Clinic",
      "note": "Specializes in acne and pigmentation treatment"
    }},
    {{
      "name": "DermCare Center",
      "note": "Offers mole checks and skin cancer screening"
    }}
  ],
  "doctors": [
    {{
      "name": "Dr. Jane Smith",
      "specialty": "Dermatologist",
      "note": "Expert in melanoma detection"
    }},
    {{
      "name": "Dr. John Doe",
      "specialty": "Skin Specialist",
      "note": "Focuses on cosmetic skin treatments"
    }}
  ]
}}
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(prompt)
        text = result.text

        # Extract JSON block from Gemini output
        start = text.find('{')
        end = text.rfind('}') + 1
        parsed = json.loads(text[start:end])

        return parsed.get("clinics", []), parsed.get("doctors", [])
    except Exception as e:
        print("❌ Gemini error:", e)
        print("❌ Raw Gemini output:", text if 'text' in locals() else '[No output]')
        return [], []

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

        # === Normalized lookup for treatment
        normalized_class = predicted_class.strip().lower()
        treatment = next(
            (v for k, v in treatments.items() if k.strip().lower() == normalized_class),
            "No treatment info available."
        )

        # === Gemini integration
        suggested_clinics, suggested_doctors = get_clinics_and_doctors(lat, lon)

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
            'suggested_clinics': suggested_clinics,
            'suggested_doctors': suggested_doctors
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500





if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Python model microservice running on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
