# === Full Backend for DR.Epidermus ===
import os
import json
import traceback
import ssl
import certifi
import pymongo
import requests
import numpy as np
import cv2

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

ssl._create_default_https_context = ssl._create_unverified_context

# === Load environment ===
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("⚠️ GEMINI_API_KEY is missing")

# === Gemini setup ===
genai.configure(api_key=gemini_api_key)

# === Flask app ===
app = Flask(__name__)
CORS(app)

# === MongoDB connection ===
MONGO_URI = "mongodb+srv://jtranberg:vhdvJR1CTc8FhdGN@cluster0.cwpequc.mongodb.net/drepidermus?retryWrites=false&w=majority&ssl=true&authSource=admin&appName=SkinScan"

def get_users_collection():
    try:
        client = MongoClient(
            MONGO_URI,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=5000
        )
        print("✅ Connected to MongoDB")
        return client['drepidermus']['users']
    except Exception as e:
        print("❌ Failed to connect to MongoDB:", e)
        return None

# === Model Setup ===
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.keras')
TRIAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model3.keras')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels_8.json')
TREATMENTS_PATH = os.path.join(MODEL_DIR, 'treatments.json')

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
triage_model = load_model(TRIAGE_MODEL_PATH) if os.path.exists(TRIAGE_MODEL_PATH) else None
class_labels = json.load(open(LABELS_PATH)) if os.path.exists(LABELS_PATH) else ["Unknown"]
treatments = json.load(open(TREATMENTS_PATH)) if os.path.exists(TREATMENTS_PATH) else {}

# === Routes Below ===

# === Auth Routes ===
@app.route('/register', methods=['POST'])
def register():
    users_collection = get_users_collection()
    if users_collection is None:
        return jsonify({"success": False, "message": "Database not connected"}), 500

    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password are required"}), 400
        if len(password) < 6:
            return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400

        if users_collection.find_one({'email': email}):
            return jsonify({"success": False, "message": "User already exists"}), 409

        users_collection.insert_one({
            'email': email,
            'password': generate_password_hash(password)
        })

        return jsonify({"success": True, "message": "Registration successful", "email": email}), 201
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": "Registration failed"}), 500


@app.route('/login', methods=['POST'])
def login():
    users_collection = get_users_collection()
    if not users_collection:
        return jsonify({'error': 'Database not connected'}), 500

    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        user = users_collection.find_one({'email': email})
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'error': 'Invalid credentials'}), 401

        return jsonify({'message': 'Login successful', 'email': email}), 200
    except Exception as e:
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
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# === Launch ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 DR.Epidermus Backend starting on port {port} in DEBUG mode...")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
