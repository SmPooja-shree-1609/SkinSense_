"""
predictor.py
------------
SkinSense — model loading, image preprocessing, and prediction logic.

Key facts about this model (verified):
- Architecture  : EfficientNet-based, trained on HAM10000 (7 skin lesion classes)
- Input         : raw float32 pixel values in [0, 255] — NO manual normalization needed
- Why           : the model has built-in Rescaling + Normalization layers as its
                  first two layers, so it handles all scaling internally
- preprocess_input is NOT used here — in Keras 2.13+ it is a documented no-op
                  for EfficientNet (returns input unchanged). Using it would add
                  confusion without any effect.
- Output        : softmax probabilities over 7 classes, shape (1, 7)
"""

import os
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "best_modelnew.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# EfficientNet input size used during training
IMG_SIZE = (224, 224)

# HAM10000 class names in alphabetical order — matches how LabelEncoder was fit
# Used as fallback when label_encoder.pkl is not available
FALLBACK_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_model():
    """
    Load the trained Keras model from disk.

    compile=False — we are only doing inference, not training.
    This skips rebuilding the optimizer and speeds up loading.

    Note: Keras 2.13 prints the full model JSON config to stdout during load
    when the model was saved with a newer Keras format. This is a cosmetic
    quirk — it does not affect functionality and can be safely ignored.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: '{MODEL_PATH}'. "
            "Make sure best_modelnew.h5 is in the project root."
        )
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


def load_label_encoder():
    """
    Load the scikit-learn LabelEncoder saved during training.

    Returns None if label_encoder.pkl is not found.
    In that case, FALLBACK_CLASSES will be used for class names.
    """
    if not os.path.exists(LABEL_ENCODER_PATH):
        return None
    with open(LABEL_ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return encoder


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepare a PIL image for model input.

    Steps:
    1. Convert to RGB  — handles grayscale (L) and transparent (RGBA) uploads
    2. Resize to 224x224 — the input size this model was trained on
    3. Convert to float32 numpy array — pixel values stay in [0, 255]
    4. Add batch dimension — shape goes from (224, 224, 3) to (1, 224, 224, 3)

    Why NOT divide by 255?
    This model includes Rescaling and Normalization as its first two internal
    layers. Dividing by 255 before passing to the model would double-normalize
    the input and produce incorrect predictions. Pass raw [0, 255] values only.

    Why NOT use preprocess_input?
    tensorflow.keras.applications.efficientnet.preprocess_input is a no-op
    in Keras 2.13 — it returns the input unchanged. It was kept for API
    compatibility but does nothing. We skip it to keep the code honest.
    """
    # Step 1: Ensure 3-channel RGB
    image = image.convert("RGB")

    # Step 2: Resize to model's expected input size
    image = image.resize(IMG_SIZE)

    # Step 3: Convert to float32 — values remain in [0, 255]
    img_array = np.array(image, dtype=np.float32)

    # Step 4: Add batch dimension → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict(image: Image.Image, model, label_encoder=None) -> dict:
    """
    Run a full prediction on a PIL Image.

    Returns a dict:
    - 'label'           : predicted class name (str)
    - 'confidence'      : top class confidence as a percentage (float)
    - 'all_scores'      : {class_name: confidence_%} for all 7 classes (dict)
    - 'image_shape'     : shape of the array fed to the model (tuple) — for debug
    - 'raw_predictions' : raw softmax output from model.predict() — for debug
    - 'predicted_index' : integer index of the top class — for debug
    """
    # Preprocess the image into the correct array format
    img_array = preprocess_image(image)

    # Run the model — this is the only source of predictions
    raw_predictions = model.predict(img_array, verbose=0)  # shape: (1, 7)
    scores = raw_predictions[0]                             # shape: (7,)

    # Find the class with the highest probability
    predicted_index = int(np.argmax(scores))
    confidence = float(scores[predicted_index]) * 100

    # Resolve class names from encoder or fallback list
    if label_encoder is not None:
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        class_names = list(label_encoder.classes_)
    else:
        class_names = FALLBACK_CLASSES
        predicted_label = class_names[predicted_index]

    # Build probability breakdown for all 7 classes
    all_scores = {
        class_names[i]: round(float(scores[i]) * 100, 2)
        for i in range(len(scores))
    }

    return {
        "label": predicted_label,
        "confidence": round(confidence, 2),
        "all_scores": all_scores,
        "image_shape": img_array.shape,
        "raw_predictions": raw_predictions.tolist(),
        "predicted_index": predicted_index,
    }
