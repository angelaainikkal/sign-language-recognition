"""
make_features.py - turn the A-Z image folders into features.npy and labels.npy

Usage:
    python make_features.py --data "C:\\Users\\Admin\\Downloads\\sign_dataset\\original_images"

Optional:
    --size 32     image is resized to size x size (default 32 -> 1024 features)
    --out .       folder where features.npy and labels.npy are saved

Images are read in numeric order (1.jpg, 2.jpg, ...) so the order matches capture time.
Each image: read -> grayscale -> resize -> scale pixels to 0-1 -> flatten.
Only folders named with a single letter A-Z are used (digit folders are skipped).

Install:  pip install opencv-python numpy
"""
import argparse
import os

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder containing A, B, C... subfolders")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--out", default=".")
    args = p.parse_args()

    letters = [d for d in sorted(os.listdir(args.data))
               if len(d) == 1 and d.isalpha() and
               os.path.isdir(os.path.join(args.data, d))]
    if not letters:
        raise SystemExit("No letter folders found. Check the --data path.")
    print(f"Found {len(letters)} letter folders: {''.join(letters)}")

    X, y, skipped = [], [], 0
    for letter in letters:
        folder = os.path.join(args.data, letter)
        count = 0
        def num_key(fn):
            stem = os.path.splitext(fn)[0]
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for name in sorted(files, key=num_key):  # 1,2,3... = capture order
            img = cv2.imread(os.path.join(folder, name))
            if img is None:  # unreadable / corrupt image
                skipped += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (args.size, args.size))
            X.append((small.astype(np.float32) / 255.0).flatten())
            y.append(letter)
            count += 1
        print(f"  {letter}: {count} images")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "features.npy"), X)
    np.save(os.path.join(args.out, "labels.npy"), y)
    print(f"\nSaved features.npy {X.shape} and labels.npy {y.shape}")
    print(f"Skipped unreadable images: {skipped}")


if __name__ == "__main__":
    main()
