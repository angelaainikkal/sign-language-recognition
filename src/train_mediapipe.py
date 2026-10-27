
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "landmarks.npz"
MODEL_FILE = ROOT / "models" / "mediapipe_svm.joblib"

# Load extracted MediaPipe landmarks
data = np.load(DATA_FILE)
X = data["X"]
y = data["L"].astype(str)

print("Dataset shape:", X.shape)
print("Number of classes:", len(np.unique(y)))

# Check that every class has enough examples for a stratified split
classes, counts = np.unique(y, return_counts=True)
print("\nSamples per class:")
for label, count in zip(classes, counts):
    print(f"{label}: {count}")

if counts.min() < 2:
    raise ValueError(
        "At least one class has fewer than 2 samples. "
        "Collect more examples before using a stratified split."
    )

# Keep the class proportions similar in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# Scale features and train an SVM classifier
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
)

model.fit(X_train, y_train)

# Evaluate on data not used for training
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MEDIAPIPE + SVM RESULTS ==========")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=classes,
        zero_division=0,
    )
)

print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred, labels=classes))
print("\nClass order:", list(classes))

# Save classifier and class labels
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "classes": classes,
        "feature_count": X.shape[1],
    },
    MODEL_FILE,
)

print(f"\nModel saved to: {MODEL_FILE}")
