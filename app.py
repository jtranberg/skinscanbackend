from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import google.generativeai as genai
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import re
from PIL import Image
import traceback
import requests
import certifi

# === Load environment variables ===
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("⚠️ Warning: GEMINI_API_KEY is missing.")

# === Gemini config ===
genai.configure(api_key=gemini_api_key)

# === Flask setup ===
app = Flask(__name__)
CORS(app)

# === MongoDB Setup ===
MONGO_URI = os.getenv("MONGO_URI") or "mongodb://localhost:27017"
client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client['epidermus']
users_collection = db['users']

try:
    client.server_info()
    print("✅ Connected to MongoDB successfully")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB: {e}")



def download_if_missing(local_path, github_url):
    if not os.path.exists(local_path):
        print(f"⬇️ Downloading {os.path.basename(local_path)}...")
        response = requests.get(github_url, allow_redirects=True)
        
        # ✅ Ensure it's a binary stream
        if response.status_code == 200 and "html" not in response.headers.get("Content-Type", ""):
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded {os.path.basename(local_path)}")
        else:
            print(f"❌ Failed to download {os.path.basename(local_path)} - Not a binary file")



# === Before loading models ===
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

download_if_missing(
    os.path.join(MODEL_DIR, 'best_model3.keras'),
    'https://github.com/jtranberg/3_class_model/releases/download/v1.0/best_model3.keras'

)

download_if_missing(
    os.path.join(MODEL_DIR, 'best_model.keras'),
    'https://github.com/jtranberg/8_class_model/releases/download/v1.0/best_model.keras'


)



# === Paths ===
MODEL_PATH = os.path.join('model', 'best_model.keras')
TRIAGE_MODEL_PATH = os.path.join('model', 'best_model3.keras')
LABELS_PATH = os.path.join('model', 'class_labels_8.json')
TREATMENTS_PATH = os.path.join('model', 'treatments.json')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load 8-class model ===
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("✅ 8-Class Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading 8-class model: {e}")
        model = None
else:
    print("❌ 8-class model file not found.")
    model = None

# === Load 3-class triage model ===
if os.path.exists(TRIAGE_MODEL_PATH):
    try:
        triage_model = load_model(TRIAGE_MODEL_PATH)
        print("✅ 3-Class Triage Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load 3-class model: {e}")
        triage_model = None
else:
    print("❌ 3-class model not found.")
    triage_model = None

# === Load class labels ===
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, 'r') as f:
        class_labels = json.load(f)
    print(f"✅ Loaded class labels: {class_labels}")
else:
    class_labels = ["Unknown"]
    print("⚠️ No class_labels_8.json found.")

# === Load treatment info ===
if os.path.exists(TREATMENTS_PATH):
    with open(TREATMENTS_PATH, 'r') as f:
        treatments = json.load(f)
    print("✅ Loaded treatment data.")
else:
    treatments = {}
    print("⚠️ No treatments.json found.")

# === OpenCV Skin Detection ===
def detect_skin(img_array):
    try:
        img_bgr = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2BGR)
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 40, 60], dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower, upper)
        skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        print(f"🧪 Skin detection: {skin_ratio:.2%} of image is skin")
        return skin_ratio > 0.35
    except Exception as e:
        print("❌ Skin detection error:", e)
        return False

# === Login Endpoint ===
@app.route('/login', methods=['POST'])
def login():
    try:
        print("🗬 Login request received")
        print("Headers:", dict(request.headers))
        print("Raw Body:", request.get_data(as_text=True))

        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        user = users_collection.find_one({'email': email})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        if not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid credentials'}), 401

        return jsonify({'message': 'Login successful', 'email': email}), 200

    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

# === Chatbot Endpoint ===
@app.route('/chatbot', methods=['POST'])
def chatbot():
    print("✅ Chatbot route hit!")

    try:
        raw_body = request.get_data(as_text=True)
        print("📨 Raw Body:", raw_body)

        # Try to parse JSON using both methods
        try:
            data = request.get_json(force=True)
            print("📦 Parsed with get_json():", data)
        except Exception as e:
            print("⚠️ get_json() failed:", e)
            data = json.loads(raw_body)
            print("📦 Fallback to json.loads():", data)

        # Ensure dict, not string
        if isinstance(data, str):
            data = json.loads(data)

        query = data.get('query', '')
        print(f"💬 Query received: {query}")

        if not query:
            return jsonify({'error': 'Prompt is required'}), 400

        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(query)

        if hasattr(result, 'text') and result.text:
            reply = result.text.strip()
        elif hasattr(result, 'parts') and result.parts:
            reply = result.parts[0].text.strip()
        else:
            reply = "🧠 Gemini gave an empty response."

        print(f"🤖 Gemini reply: {reply}")
        return jsonify({'response': reply})

    except Exception as e:
        print("❌ Gemini Chatbot Exception:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# === Predict Endpoint ===
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

        if triage_model:
            triage_result = triage_model.predict(img_array)[0]
            triage_index = np.argmax(triage_result)
            triage_class = ["Normal", "Benign", "Malignant"][triage_index]
        else:
            triage_class = "Unknown"

        if triage_class == "Normal":
            predicted_class = "Normal"
            confidence = 1.0
            top1 = top2 = top3 = "Normal"
        else:
            if model:
                prediction_probs = model.predict(img_array)[0]
                top_indices = prediction_probs.argsort()[-3:][::-1]
                top_labels = [class_labels[i] for i in top_indices]
                top_probs = [float(prediction_probs[i]) for i in top_indices]
                predicted_class = top_labels[0]
                confidence = top_probs[0]
                top1, top2, top3 = top_labels
            else:
                predicted_class = "Unknown"
                confidence = 0.0
                top1 = top2 = top3 = "Unknown"

        # Treatment and location (Gemini)
        treatment = treatments.get(predicted_class, "No treatment info available.")

        prompt = f"""
You are an AI assistant for a skin diagnosis app. The user has been diagnosed with {predicted_class}.
Generate a plausible JSON response with fictional but realistic examples of clinics and doctors near coordinates:
Latitude: {lat}, Longitude: {lon}.

Respond **only** in JSON format like:
{{
  "clinics": [
    {{
      "name": "Pacific Dermatology Center",
      "address": "123 Skin Ave, Victoria, BC",
      "note": "Specialist in {predicted_class}"
    }}
  ],
  "doctors": [
    {{
      "name": "Dr. Eva Tran",
      "specialty": "{predicted_class}",
      "note": "Board-certified dermatologist with experience in skin cancer"
    }}
  ]
}}
"""

        model_gen = genai.GenerativeModel("gemini-1.5-flash")
        result = model_gen.generate_content(prompt)
        print("🌍 Gemini Geo Response:", result.text)

        # ✅ Strip code block wrappers if present
        try:
            raw_text = result.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            geo_json = json.loads(raw_text)
        except Exception as e:
            print("❌ Failed to parse Gemini JSON:", e)
            geo_json = {"clinics": [], "doctors": []}

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "top1": top1,
            "top2": top2,
            "top3": top3,
            "age": age,
            "gender": gender,
            "weight": weight,
            "most_common_treatment": treatment,
            "suggested_clinics": geo_json.get("clinics", []),
            "suggested_doctors": geo_json.get("doctors", [])
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


        model_gen = genai.GenerativeModel("gemini-1.5-flash")
        result = model_gen.generate_content(prompt)
        print("🌍 Gemini Geo Response:", result.text)

        try:
            geo_json = json.loads(result.text)
        except:
            geo_json = {"clinics": [], "doctors": []}

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "top1": top1,
            "top2": top2,
            "top3": top3,
            "age": age,
            "gender": gender,
            "weight": weight,
            "most_common_treatment": treatment,
            "suggested_clinics": geo_json.get("clinics", []),
            "suggested_doctors": geo_json.get("doctors", [])
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Inject test user
if users_collection.count_documents({'email': 'jtranberg@hotmail.com'}) == 0:
    users_collection.insert_one({
        'email': 'jtranberg@hotmail.com',
        'password': generate_password_hash('sa')
    })
    print("✅ Test user inserted into MongoDB")

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render will provide a dynamic port
    print(f"🚀 DR.Epidermus Backend starting on port {port} in DEBUG mode...")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)

