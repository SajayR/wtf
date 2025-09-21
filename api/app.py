#!/usr/bin/env python3
"""Flask API for predictions, feedback capture, and retraining trigger."""

import base64
import imghdr
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Tuple
from uuid import uuid4

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

# DO NOT import db here. This was the source of the circular import.
# from . import db 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = Path(os.getenv("MODEL_DIR", "ml/artifacts/current"))
WEAK_DIR = Path(os.getenv("WEAK_FEEDBACK_DIR", "data/weak_feedback"))
THRESHOLD_FILE = MODEL_DIR / "threshold.txt"
MODEL_FILE = MODEL_DIR / "model.keras"
LABEL_MAP_FILE = MODEL_DIR / "label_map.json"
RETRAIN_SCRIPT = Path(os.getenv("RETRAIN_SCRIPT", "ml/retrain.py"))

# --- Globals for the model (managed by the factory) ---
model_lock = threading.Lock()
model = None
threshold = 0.5
inv_label_map: Dict[int, str] = {0: "no", 1: "yes"}
CLASS_MAP: Dict[str, int] = {"no": 0, "yes": 1}
IMG_SIZE = 224

def _load_label_map() -> None:
    """Load label map from file if available."""
    global inv_label_map, CLASS_MAP
    if LABEL_MAP_FILE.exists():
        with LABEL_MAP_FILE.open() as fh:
            loaded = json.load(fh)
        inv_label_map = {int(k): v for k, v in loaded.items()}
        CLASS_MAP = {v: k for k, v in inv_label_map.items()}
    else:
        inv_label_map = {0: "no", 1: "yes"}
        CLASS_MAP = {"no": 0, "yes": 1}


def load_model_artifacts() -> None:
    """Load model and threshold from artifacts."""
    global model, threshold
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_FILE}. Run training first.")
    
    # --- THIS IS THE FIX FOR THE *NEXT* ERROR ---
    # We must import the custom function your model uses
    from ml.train import preprocess_input
    
    with model_lock:
        # And we must tell Keras about it
        custom_objects = {'preprocess_input': preprocess_input}
        model = tf.keras.models.load_model(MODEL_FILE, custom_objects=custom_objects)
        
        if THRESHOLD_FILE.exists():
            threshold = float(THRESHOLD_FILE.read_text().strip())
        _load_label_map()


def ensure_dirs() -> None:
    """Ensure weak feedback directories exist."""
    for cls in CLASS_MAP:
        (WEAK_DIR / cls).mkdir(parents=True, exist_ok=True)


def decode_base64_image(data: str) -> bytes:
    """Decode base64-encoded image string into bytes."""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


def image_bytes_to_tensor(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into model-ready numpy array."""
    from ml.train import preprocess_input
    
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    arr = img.numpy()
    return np.expand_dims(arr, axis=0)


def write_feedback_image(image_bytes: bytes, label: str) -> Path:
    """Save feedback image under the given label directory."""
    ext = imghdr.what(None, image_bytes) or "png"
    if ext == "jpeg":
        ext = "jpg"
    fname = f"fb_{int(time.time())}_{uuid4().hex[:8]}.{ext}"
    dest = WEAK_DIR / label / fname
    dest.write_bytes(image_bytes)
    return dest

# --- Application Factory ---

def create_app(*args, **kwargs) -> Flask:
    """Application factory for Flask."""
    
    app = Flask(__name__) # Create the app instance HERE
    
    # --- THIS IS THE FIX FOR THE CIRCULAR IMPORT ---
    # We import db *inside* the factory, not at the top of the file
    from . import db

    # --- Register Routes with the app instance ---
    
    @app.route("/health", methods=["GET"])
    def health() -> Tuple[str, int]:
        return jsonify({"status": "ok", "model_loaded": model is not None}), 200

    @app.route("/predict", methods=["POST"])
    def predict() -> Tuple[str, int]:
        payload = request.get_json(force=True)
        if not payload or "image" not in payload:
            return jsonify({"error": "missing image"}), 400

        if model is None:
            return jsonify({"error": "model not loaded"}), 503

        image_bytes = decode_base64_image(payload["image"])
        arr = image_bytes_to_tensor(image_bytes)
        with model_lock:
            if model is None:
                return jsonify({"error": "model not loaded"}), 503
            probs = model.predict(arr, verbose=0).flatten()
        
        prob = float(probs[0])
        label_idx = int(prob >= threshold)
        label = inv_label_map.get(label_idx, str(label_idx))
        db.log_prediction(probability=prob, predicted_label=label, threshold=threshold)
        return jsonify({
            "probability": prob,
            "prediction": label,
            "threshold": threshold,
        }), 200

    @app.route("/feedback", methods=["POST"])
    def feedback() -> Tuple[str, int]:
        payload = request.get_json(force=True)
        required = {"image", "correct_label", "predicted_label"}
        if not payload or not required.issubset(payload):
            return jsonify({"error": "missing fields"}), 400

        correct_label = payload["correct_label"].lower()
        predicted_label = payload["predicted_label"].lower()
        if correct_label not in CLASS_MAP:
            return jsonify({"error": f"unknown label {correct_label}"}), 400

        image_bytes = decode_base64_image(payload["image"])
        dest = write_feedback_image(image_bytes, correct_label)
        probability = float(payload.get("probability", 0.0))
        db.store_feedback(
            predicted_label=predicted_label,
            correct_label=correct_label,
            probability=probability,
            image_path=str(dest),
        )
        return jsonify({"status": "stored", "path": str(dest)}), 200

    @app.route("/trigger-retrain", methods=["POST"])
    def trigger_retrain() -> Tuple[str, int]:
        app.logger.info("Received request to reload model artifacts.")
        try:
            load_model_artifacts()
            app.logger.info("Successfully reloaded model artifacts.")
            return jsonify({"status": "model_reloaded"}), 200
        except FileNotFoundError as exc:
            app.logger.warning("Failed to reload artifacts: %s", exc)
            return jsonify({"error": "model artifacts not found, training may still be in progress"}), 503
        except Exception as exc:
            # Add exc_info=True to log the full traceback
            app.logger.error("An unexpected error occurred during model reload: %s", exc, exc_info=True)
            return jsonify({"error": str(exc)}), 500

    # --- Initialization ---
    with app.app_context():
        ensure_dirs()
        db.init_db()
        try:
            load_model_artifacts()
            app.logger.info("Model loaded successfully on startup.")
        except FileNotFoundError as exc:
            app.logger.warning("model not loaded on startup: %s", exc)
        except Exception as exc:
            # Add exc_info=True to log the full traceback
            app.logger.error("Failed to load model on startup: %s", exc, exc_info=True)
    
    return app