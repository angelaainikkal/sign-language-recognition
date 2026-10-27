
from pathlib import Path
import argparse
import io

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "mediapipe_svm.joblib"
HAND_MODEL_FILE = ROOT / "models" / "hand_landmarker.task"


def predict(image_bytes: bytes) -> dict:
    """Predict a hand-sign alphabet from uploaded image bytes."""

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb_image = np.array(pil_image)
    except Exception:
        return {
            "status": "invalid_file",
            "message": "Please upload a valid image.",
        }

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Classifier not found: {MODEL_FILE}")

    if not HAND_MODEL_FILE.exists():
        raise FileNotFoundError(f"Hand landmark model not found: {HAND_MODEL_FILE}")

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(HAND_MODEL_FILE)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return {
            "status": "no_hand",
            "message": "No hand detected. Try a clear image with the whole hand visible.",
        }

    hand = result.hand_landmarks[0]

    # Normalize exactly as during training.
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32,
    )
    points -= points[0]

    scale = np.max(np.abs(points))
    if scale > 0:
        points /= scale

    features = points.flatten().reshape(1, -1)

    if features.shape[1] != bundle["feature_count"]:
        raise ValueError("Extracted landmark feature count does not match the model.")

    prediction = str(model.predict(features)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

    # Draw landmark points and connections over the uploaded image.
    annotated = rgb_image.copy()
    height, width = annotated.shape[:2]

    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    for connection in connections:
        start = hand[connection.start]
        end = hand[connection.end]
        p1 = (int(start.x * width), int(start.y * height))
        p2 = (int(end.x * width), int(end.y * height))
        cv2.line(
            annotated,
            p1,
            p2,
            (90, 220, 120),
            2,
            cv2.LINE_AA,
        )

    for landmark in hand:
        center = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(
            annotated,
            center,
            4,
            (255, 100, 80),
            -1,
            cv2.LINE_AA,
        )

    return {
        "status": "ok",
        "label": prediction,
        "confidence": confidence,
        "annotated_image": annotated,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict a sign alphabet using MediaPipe + SVM."
    )
    parser.add_argument("image", help="Path to a test image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = predict(image_path.read_bytes())

    if result["status"] == "ok":
        print(f"Image: {image_path.name}")
        print(f"Predicted sign: {result['label']}")
        if result["confidence"] is not None:
            print(f"Classifier score: {result['confidence']:.2%}")
    else:
        print(result["message"])


if __name__ == "__main__":
    main()
