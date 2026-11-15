"""Predict the pest class of one image using a saved model (no training)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import IMAGE_SIZE, MODELS_DIR


def predict_image(image_path: Path, model_path: Path = None):
    image_path = Path(image_path).expanduser()
    model_path = model_path or MODELS_DIR / "pest_resnet152v2_final.keras"
    labels_path = MODELS_DIR / "class_names.json"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Train once or obtain the "
            "model file from your teammate."
        )
    if not labels_path.is_file():
        raise FileNotFoundError(f"Class label file not found: {labels_path}")
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(IMAGE_SIZE)
            image_array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(f"Cannot read image {image_path}: {error}") from error

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "Prediction requires TensorFlow in the active Python environment. "
            "Install the dependencies from requirements.txt."
        ) from error
    with labels_path.open(encoding="utf-8") as file:
        class_names = json.load(file)
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise ValueError(
            "The number of model outputs does not match class_names.json."
        )
    order = np.argsort(probabilities)[::-1]
    result = {
        "predicted_class": class_names[int(order[0])],
        "confidence_percent": float(probabilities[order[0]] * 100),
        "probabilities_percent": {
            class_names[int(index)]: float(probabilities[index] * 100)
            for index in order
        },
    }
    print(f"Predicted pest: {result['predicted_class']}")
    print(f"Confidence: {result['confidence_percent']:.2f}%")
    print("Class probabilities:")
    for class_name, confidence in result["probabilities_percent"].items():
        print(f"  {class_name}: {confidence:.2f}%")
    return result


def choose_image_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError(
            "The image picker requires Tkinter. Run the script with a standard "
            "Python installation, or pass an image path directly."
        ) from error

    try:
        window = tk.Tk()
        window.withdraw()
    except tk.TclError as error:
        raise RuntimeError(
            "Could not open the image picker. Pass the image path as an argument."
        ) from error

    try:
        selected_file = filedialog.askopenfilename(
            parent=window,
            title="Choose a pest image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
    finally:
        window.destroy()
    return Path(selected_file) if selected_file else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="Path to an image; omit this to choose an image in a file dialog",
    )
    parser.add_argument("--model", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        image_path = args.image or choose_image_file()
        if image_path is None:
            print("No image selected; prediction cancelled.")
        else:
            predict_image(image_path, args.model)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
