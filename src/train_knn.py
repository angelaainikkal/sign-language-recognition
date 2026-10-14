
from pathlib import Path
import json
import time

import cv2
import joblib
import numpy as np

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. Load the processed dataset
# --------------------------------------------------
print("Loading dataset...")

images = np.load(DATA_DIR / "images.npy")
labels = np.load(DATA_DIR / "labels.npy")

print("Original images:", images.shape)
print("Original labels:", labels.shape)

# --------------------------------------------------
# 3. Keep alphabet classes A-Z only
# --------------------------------------------------
alphabet_mask = np.isin(labels, list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

images = images[alphabet_mask]
labels = labels[alphabet_mask]

print("Alphabet images:", len(images))
print("Alphabet classes:", len(np.unique(labels)))

# --------------------------------------------------
# 4. Extract HOG features
# --------------------------------------------------
print("\nExtracting HOG features...")
start = time.time()

features = []

for i, image in enumerate(images):
    image = image.astype(np.uint8)

    feature = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    features.append(feature)

    if (i + 1) % 2000 == 0:
        print(f"Processed {i + 1}/{len(images)} images")

X = np.asarray(features, dtype=np.float32)
y = labels

print("Feature matrix:", X.shape)
print(f"Feature extraction time: {time.time() - start:.1f} seconds")

# --------------------------------------------------
# 5. Split into training and testing sets
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 6. Train and compare KNN configurations
# --------------------------------------------------
results = []
best_model = None
best_k = None
best_accuracy = -1

for k in [3, 5, 7, 9]:
    print(f"\nTraining KNN with K={k}...")
    start = time.time()

    model = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "k": k,
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Elapsed:   {time.time() - start:.1f} seconds")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_k = k
        best_predictions = predictions

# --------------------------------------------------
# 7. Save results and best model
# --------------------------------------------------
model_path = MODEL_DIR / "knn_hog_alphabet.joblib"
results_path = MODEL_DIR / "knn_results.json"

joblib.dump(
    {
        "model": best_model,
        "feature_method": "HOG",
        "image_size": [64, 64],
        "classes": sorted(np.unique(y).tolist()),
        "best_k": best_k,
    },
    model_path,
)

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nRESULTS SUMMARY")
print("----------------")
for result in results:
    print(
        f"K={result['k']} | "
        f"Accuracy={result['accuracy']:.4f} | "
        f"F1={result['f1_macro']:.4f}"
    )

print(f"\nBest K by test accuracy: {best_k}")
print(f"Best test accuracy: {best_accuracy:.4f}")

print("\nClassification report for the selected model:")
print(classification_report(y_test, best_predictions, zero_division=0))

print("Saved model:", model_path)
print("Saved results:", results_path)
