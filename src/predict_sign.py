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

# Reject a letter if the SVM is less sure than this.
# With 27 classes, correct answers often score 0.3-0.6, so tune this on
# photos the model has NOT seen (not on training images).
CONF_THRESHOLD = 0.40
NOT_A_SIGN = "not_a_sign"
MESSAGE = "This doesn't look like a hand sign"

_bundle = None


def _load():
    """Load the model once. Keys: kmeans, svm, classes."""
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Pull the latest sift-svm branch (models/sift_svm.joblib)."
            )
        bundle = joblib.load(MODEL_PATH)
        missing = {"kmeans", "svm", "classes"} - set(bundle)
        if missing:
            raise KeyError(f"Model file is missing keys: {sorted(missing)}")
        _bundle = bundle
    return _bundle


def _reject(reason, conf=0.0, best_guess=None):
    return {
        "status": "rejected",
        "reason": reason,
        "message": MESSAGE,
        "confidence": float(conf),
        "best_guess": best_guess,
    }


def predict(image_bytes: bytes) -> dict:
    """Used by the upload and the webcam. Takes raw image bytes.

    Returns:
      {"status": "ok", "label": "B", "confidence": 0.82}
      {"status": "rejected", "reason": "not_a_sign" | "low_confidence" |
       "invalid_file" | "no_features", "message": ..., "confidence": ...,
       "best_guess": "B" or None}
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return _reject("invalid_file")

    b = _load()
    hist = image_to_features(img, b["kmeans"])

    # No SIFT keypoints (blank or very blurry image): nothing to classify.
    if not np.any(hist):
        return _reject("no_features")

    svm = b["svm"]
    probs = svm.predict_proba(hist.reshape(1, -1))[0]
    i = int(np.argmax(probs))
    label = str(svm.classes_[i])   # labels are the letter strings themselves
    conf = float(probs[i])

    if label == NOT_A_SIGN:
        return _reject("not_a_sign", conf)

    if conf < CONF_THRESHOLD:
        return _reject("low_confidence", conf, best_guess=label)

    return {"status": "ok", "label": label, "confidence": conf}


def predict_sign(image_path):
    """Old interface: returns the letter, or None if the image is rejected."""
    result = predict(Path(image_path).read_bytes())
    return result["label"] if result["status"] == "ok" else None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py src/predict_sign.py <image_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = Path.cwd() / path
    print(predict(path.read_bytes()))