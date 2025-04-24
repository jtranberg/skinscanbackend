from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from PIL import Image
from geopy.geocoders import Nominatim
import google.generativeai as genai
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
import os
import json
import requests
import certifi
import traceback
import re
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
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
MONGO_URI = os.getenv("MONGO_URI")
try:
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000
    )
    db = client.get_database()
    users_collection = db['users']
    client.admin.command("ping")
    print("✅ MongoDB connected successfully.")
except Exception as e:
    print("❌ MongoDB connection failed:", e)
    users_collection = None

# === Utility: Model Downloader ===
def download_if_missing(local_path, github_url):
    if not os.path.exists(local_path):
        print(f"⬇️ Downloading {os.path.basename(local_path)}...")
        response = requests.get(github_url, allow_redirects=True)
        if response.status_code == 200 and "html" not in response.headers.get("Content-Type", ""):
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Downloaded {os.path.basename(local_path)}")
        else:
            print(f"❌ Failed to download {os.path.basename(local_path)} - Not a binary file")

# === Model Download & Load ===
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

download_if_missing(os.path.join(MODEL_DIR, 'best_model3.keras'),
                    'https://github.com/jtranberg/3_class_model/releases/download/v1.0/best_model3.keras')
download_if_missing(os.path.join(MODEL_DIR, 'best_model.keras'),
                    'https://github.com/jtranberg/8_class_model/releases/download/v1.0/best_model.keras')

MODEL_PATH = os.path.join('model', 'best_model.keras')
TRIAGE_MODEL_PATH = os.path.join('model', 'best_model3.keras')
LABELS_PATH = os.path.join('model', 'class_labels_8.json')
TREATMENTS_PATH = os.path.join('model', 'treatments.json')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
triage_model = load_model(TRIAGE_MODEL_PATH) if os.path.exists(TRIAGE_MODEL_PATH) else None
class_labels = json.load(open(LABELS_PATH)) if os.path.exists(LABELS_PATH) else ["Unknown"]
treatments = json.load(open(TREATMENTS_PATH)) if os.path.exists(TREATMENTS_PATH) else {}

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

# === Auth Routes ===
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        print("📩 Register Payload:", data)
        email = data.get('email')
        password = data.get('password')

        if email:
            email = email.strip().lower()
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        if users_collection.find_one({'email': email}):
            return jsonify({'error': 'User already exists'}), 409

        users_collection.insert_one({
            'email': email,
            'password': generate_password_hash(password)
        })

        print(f"✅ Registered user: {email}")
        return jsonify({'message': 'Registration successful', 'email': email}), 201

    except Exception as e:
        print(f"❌ Registration error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print("📩 Login Payload:", data)
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        user = users_collection.find_one({'email': email})
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        return jsonify({'message': 'Login successful', 'email': email}), 200
    except Exception as e:
        print(f"❌ Login error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Login failed'}), 500

# === Chatbot Route ===
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json(force=True)
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'Prompt is required'}), 400
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(query)
        reply = result.text.strip() if hasattr(result, 'text') else "Gemini returned no output."
        return jsonify({'response': reply})
    except Exception as e:
        print("❌ Gemini Chatbot Exception:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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

        prompt = f"""
You are an AI assistant for a skin diagnosis app. The user has been diagnosed with {predicted_class}.
Generate a plausible JSON response with fictional but realistic examples of clinics and doctors near coordinates:
Latitude: {lat}, Longitude: {lon}.

Respond **only** in JSON format like:
{{
  "clinics": [{{"name": "Clinic Name", "address": "Address", "note": "Specialty info"}}],
  "doctors": [{{"name": "Dr. Name", "specialty": "Specialty", "note": "Experience"}}]
}}
"""

        model_gen = genai.GenerativeModel("gemini-1.5-flash")
        result = model_gen.generate_content(prompt)
        print("🌍 Gemini Geo Response:", result.text)

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

# === Launch ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 DR.Epidermus Backend starting on port {port} in DEBUG mode...")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
