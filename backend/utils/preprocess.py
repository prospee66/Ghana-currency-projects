"""
Image preprocessing utilities for Ghana Currency Recognition.
These helpers are shared between train.py and app.py.
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance

IMG_SIZE = (224, 224)


# ── Core loaders ──────────────────────────────────────────────────────────────

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load an image from raw bytes and return an RGB numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def preprocess_for_model(image_bytes: bytes, target_size=IMG_SIZE) -> np.ndarray:
    """
    Standard pipeline used by the Flask API:
      load → resize → normalise → add batch dim
    Returns shape (1, H, W, 3) float32 in [0, 1].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Enhancement helpers ───────────────────────────────────────────────────────

def apply_clahe(image_np: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to improve
    appearance under varying lighting conditions.
    Input/output: RGB uint8 numpy array.
    """
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    lab      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq     = clahe.apply(l)
    lab_eq   = cv2.merge([l_eq, a, b])
    bgr_eq   = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB)


def enhance_image(
    image_np: np.ndarray,
    brightness: float = 1.0,
    contrast:   float = 1.2,
) -> np.ndarray:
    """Adjust brightness and contrast using PIL Enhance."""
    img = Image.fromarray(image_np)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return np.array(img)


# ── Region detection ──────────────────────────────────────────────────────────

def detect_note_region(image_np: np.ndarray) -> np.ndarray:
    """
    Attempt to detect and crop the currency-note region using contour detection.
    Falls back to the full image if no significant contour is found.
    """
    gray    = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_np

    largest = max(contours, key=cv2.contourArea)
    h, w    = image_np.shape[:2]

    if cv2.contourArea(largest) > 0.1 * h * w:
        x, y, cw, ch = cv2.boundingRect(largest)
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)
        return image_np[y1:y2, x1:x2]

    return image_np


# ── Full pipeline ─────────────────────────────────────────────────────────────

def full_pipeline(
    image_bytes: bytes,
    target_size: tuple = IMG_SIZE,
    use_clahe:   bool  = True,
) -> np.ndarray:
    """
    Enhanced preprocessing pipeline:
      load → CLAHE (optional) → resize → normalise → batch dim
    Returns shape (1, H, W, 3) float32 in [0, 1].
    """
    img_np = load_image_from_bytes(image_bytes)

    if use_clahe:
        img_np = apply_clahe(img_np)

    img_pil = Image.fromarray(img_np).resize(target_size, Image.LANCZOS)
    arr     = np.array(img_pil, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
