
from pathlib import Path
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "extra" / "signs"
MODEL_FILE = ROOT / "models" / "hand_landmarker.task"
OUTPUT_FILE = ROOT / "data" / "landmarks.npz"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_landmarks(landmarks):
    """Return 21 hand landmarks as 63 normalized values."""
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )

    # Make the wrist the origin.
    points -= points[0]

    # Normalize scale.
    scale = np.max(np.abs(points))
    if scale > 0:
        points /= scale

    return points.flatten()


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Hand Landmarker model not found: {MODEL_FILE}"
        )

    features = []
    labels = []
    total_images = 0
    skipped = []

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_FILE)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for class_dir in sorted(DATASET_DIR.iterdir()):
            if not class_dir.is_dir():
                continue

            label = class_dir.name.upper()
            if len(label) != 1 or not label.isalpha():
                continue

            count = 0

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                total_images += 1
                image = cv2.imread(str(image_path))

                if image is None:
                    skipped.append((image_path.name, "Cannot read image"))
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )

                result = landmarker.detect(mp_image)

                if not result.hand_landmarks:
                    skipped.append((f"{label}/{image_path.name}", "No hand detected"))
                    continue

                vector = normalize_landmarks(result.hand_landmarks[0])

                features.append(vector)
                labels.append(label)
                count += 1

            print(f"{label}: extracted {count} images")

    if not features:
        raise RuntimeError(
            "No landmarks extracted. Check the images and model file."
        )

    X = np.asarray(features, dtype=np.float32)
    L = np.asarray(labels, dtype=str)

    np.savez_compressed(OUTPUT_FILE, X=X, L=L)

    print("\n--- Extraction Summary ---")
    print(f"Images found:      {total_images}")
    print(f"Images extracted:  {len(X)}")
    print(f"Images skipped:    {len(skipped)}")
    print(f"Feature shape:     {X.shape}")
    print(f"Label shape:       {L.shape}")
    print(f"Labels:            {sorted(set(L.tolist()))}")
    print(f"Saved to:          {OUTPUT_FILE}")

    if skipped:

     print("\nSkipped images by reason:")
    print(Counter(reason for _, reason in skipped))

    print("\nSkipped images by class:")
    print(Counter(path.split("/")[0] for path, _ in skipped))

    print("\nFirst 20 skipped images:")
    for path, reason in skipped[:20]:
        print(f"- {path}: {reason}")


if __name__ == "__main__":
    main()
