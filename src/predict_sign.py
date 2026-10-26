from pathlib import Path
import sys

import cv2
import joblib
import numpy as np

# Works both as "from src.predict_sign import ..." and "python src/predict_sign.py"
try:
    from src.sift_features import image_to_features
except ModuleNotFoundError:
    from sift_features import image_to_features


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "sift_svm.joblib"

CONF_THRESHOLD = 0.40
MESSAGE = "This doesn't look like a hand sign"

_bundle = None


def _load():
    """Load the model once. Keys: kmeans, svm, classes."""
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Pull the latest sift-svm branch (Person 2 pushes it)."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _reject(reason, conf=0.0):
    return {
        "status": "rejected",
        "reason": reason,
        "message": MESSAGE,
        "confidence": float(conf),
    }


def predict(image_bytes: bytes) -> dict:
    """Used by the upload and the webcam. Takes raw image bytes.

    Returns:
      {"status": "ok", "label": "B", "confidence": 0.82}
      {"status": "rejected", "reason": ..., "message": ..., "confidence": ...}
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return _reject("invalid_file")

    b = _load()
    hist = image_to_features(img, b["kmeans"]).reshape(1, -1)
    svm, classes = b["svm"], list(b["classes"])

    probs = svm.predict_proba(hist)[0]
    i = int(np.argmax(probs))
    cls = svm.classes_[i]
    # classes_ may hold integer indexes or the label strings themselves
    label = classes[int(cls)] if isinstance(cls, (int, np.integer)) else str(cls)
    conf = float(probs[i])

    if label == "not_a_sign":
        return _reject("not_a_sign", conf)
    if conf < CONF_THRESHOLD:
        return _reject("low_confidence", conf)
    return {"status": "ok", "label": label, "confidence": conf}


def predict_sign(image_path):
    """Old interface: returns the letter, or None if the image is rejected."""
    data = Path(image_path).read_bytes()
    result = predict(data)
    return result["label"] if result["status"] == "ok" else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py src/predict_sign.py <image_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = Path.cwd() / path
    print(predict(path.read_bytes()))