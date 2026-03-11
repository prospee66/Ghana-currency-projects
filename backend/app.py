"""
Ghana Currency Recognition - Flask Backend API  (PyTorch)
Receives an uploaded image and returns the predicted denomination.
"""

import os
import io
import json
import base64

import numpy as np
import torch
from torchvision import transforms

# Limit CPU threads so the PC doesn't overheat during model load/inference
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "model", "ghana_currency_model.pt")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

model       = None
class_names = []

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif", "webp"}

# ImageNet normalisation — must match what was used during training
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Resource loading ──────────────────────────────────────────────────────────
def load_resources():
    global model, class_names

    # Class names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
    else:
        class_names = [
            "1_GHS", "2_GHS", "5_GHS", "10_GHS",
            "20_GHS", "50_GHS", "100_GHS", "200_GHS",
        ]

    # PyTorch TorchScript model
    if os.path.exists(MODEL_PATH):
        try:
            model = torch.jit.load(MODEL_PATH, map_location="cpu")
            model.eval()
            print(f"[INFO] Model loaded from {MODEL_PATH}")
        except Exception as exc:
            print(f"[ERROR] Could not load model: {exc}")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}. "
              "Run: python backend/model/train.py")


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Load image bytes and return a (1, 3, 224, 224) tensor."""
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = IMG_TRANSFORM(img).unsqueeze(0)
    return tensor


def build_response(probs: np.ndarray) -> dict:
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    prediction = class_names[pred_idx]

    all_preds = sorted(
        [
            {
                "denomination": class_names[i].replace("_", " "),
                "raw_key":      class_names[i],
                "confidence":   float(probs[i]),
                "percentage":   round(float(probs[i]) * 100, 2),
            }
            for i in range(len(class_names))
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "success":               True,
        "prediction":            prediction.replace("_", " "),
        "raw_key":               prediction,
        "confidence":            confidence,
        "confidence_percentage": round(confidence * 100, 2),
        "all_predictions":       all_preds,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Ghana Currency Recognition API",
        "version": "1.0.0",
        "backend": "PyTorch",
        "endpoints": {
            "GET  /health":  "Check API and model status",
            "POST /predict": "Upload image for denomination prediction",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "healthy",
        "model_loaded": model is not None,
        "backend":      "PyTorch",
        "classes":      class_names,
        "num_classes":  len(class_names),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": (
                "Model not loaded. "
                "Train the model first: python backend/model/train.py"
            )
        }), 503

    # ── Read image bytes ──────────────────────────────────────────────────────
    image_bytes = None

    if "file" in request.files:
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "No file selected."}), 400
        if not allowed_file(f.filename):
            return jsonify({
                "error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"
            }), 400
        image_bytes = f.read()

    elif request.is_json and "image" in request.json:
        try:
            b64 = request.json["image"]
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)
        except Exception as exc:
            return jsonify({"error": f"Invalid base64 image: {exc}"}), 400

    else:
        return jsonify({
            "error": (
                "No image provided. "
                'Send multipart/form-data with key "file", '
                'or JSON with key "image" (base64).'
            )
        }), 400

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].numpy()
        return jsonify(build_response(probs))
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(413)
def request_entity_too_large(_):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_resources()
    app.run(debug=False, host="0.0.0.0", port=5000)
