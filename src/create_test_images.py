
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

images = np.load(DATA_DIR / "images.npy")
labels = np.load(DATA_DIR / "labels.npy")

# Create one test image for every alphabet A-Z
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    indices = np.where(labels == letter)[0]

    if len(indices) == 0:
        print(f"WARNING: No image found for {letter}")
        continue

    image = images[indices[0]]

    # Handle images stored with a single channel dimension
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)

    output_path = DATA_DIR / f"test_sign_{letter}.png"

    if cv2.imwrite(str(output_path), image):
        print(f"Created: test_sign_{letter}.png")
    else:
        print(f"ERROR: Could not save image for {letter}")

print("Finished creating alphabet test images.")
