
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
from skimage.feature import hog


# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "knn_hog_alphabet.joblib"
DATA_PATH = ROOT / "data"


def extract_hog_features(image):
    """Extract HOG features from a grayscale image."""
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )


def load_model():
    """Load the trained KNN model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Please run py src/train_knn.py first."
        )

    saved = joblib.load(MODEL_PATH)

    # The training script saves the model inside a dictionary.
    if isinstance(saved, dict):
        return saved["model"]

    return saved


def predict_sign(image_path):
    """Predict the alphabet letter from an image."""
    model = load_model()

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Match the image size used during training.
    image = cv2.resize(image, (64, 64))

    features = extract_hog_features(image).reshape(1, -1)
    prediction = model.predict(features)[0]

    return prediction


def save_sample_image():
    """Save one alphabet A image from the preprocessed dataset."""
    images_path = DATA_PATH / "images.npy"
    labels_path = DATA_PATH / "labels.npy"

    images = np.load(images_path)
    labels = np.load(labels_path)

    # Find the first image labelled A.
    indices = np.where(labels == "A")[0]

    if len(indices) == 0:
        raise ValueError("No image labelled A was found in the dataset.")

    index = indices[0]
    output_path = DATA_PATH / "test_sign_A.png"

    success = cv2.imwrite(str(output_path), images[index])

    if not success:
        raise OSError("Could not save the sample image.")

    print(f"Sample image saved: {output_path}")
    return output_path


if __name__ == "__main__":
    try:
        if len(sys.argv) >= 2:
            # Predict an image supplied in the terminal.
            image_path = Path(sys.argv[1])

            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            letter = predict_sign(image_path)
            print(f"\nPredicted sign: {letter}")

        else:
            # No image supplied: create a sample and predict it.
            print("No image provided. Creating a sample image...")
            sample_path = save_sample_image()

            letter = predict_sign(sample_path)
            print(f"\nExpected sign: A")
            print(f"Predicted sign: {letter}")

    except (ValueError, FileNotFoundError, KeyError, OSError) as error:
        print(f"\nError: {error}")
