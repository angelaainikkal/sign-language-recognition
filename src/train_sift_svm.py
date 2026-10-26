
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "features_sift.npz"
KMEANS_PATH = ROOT / "models" / "sift_kmeans.joblib"
MODEL_PATH = ROOT / "models" / "sift_svm.joblib"
RESULTS_PATH = ROOT / "models" / "sift_svm_results.json"

RANDOM_STATE = 42


def load_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")
    if not KMEANS_PATH.exists():
        raise FileNotFoundError(f"KMeans vocabulary not found: {KMEANS_PATH}")

    data = np.load(FEATURES_PATH, allow_pickle=True)
    required = {"X", "y", "source", "frame", "path"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Missing feature arrays: {sorted(missing)}")

    X = data["X"].astype(np.float32)
    y = data["y"].astype(str)
    source = data["source"].astype(str)
    frame = data["frame"].astype(int)
    paths = data["path"].astype(str)

    if not (len(X) == len(y) == len(source) == len(frame) == len(paths)):
        raise ValueError("Feature arrays have inconsistent lengths.")
    if X.ndim != 2 or X.shape[1] != 200:
        raise ValueError(f"Expected X shape (N, 200), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values.")

    kmeans = joblib.load(KMEANS_PATH)
    if X.shape[1] != kmeans.n_clusters:
        raise ValueError(
            f"Feature width {X.shape[1]} does not match "
            f"KMeans clusters {kmeans.n_clusters}."
        )

    return X, y, source, frame, paths, kmeans


def make_splits(y, source, frame):
    """Keep dataset test frames separate from tuning and training."""
    dataset = source == "dataset"

    train_mask = dataset & (frame < 600)
    val_mask = dataset & (frame >= 600) & (frame < 750)
    test_mask = dataset & (frame >= 800)

    # Additional real photos and non-sign images do not have dataset
    # frame numbers. Split these separately into train and validation.
    extra_indices = np.flatnonzero(~dataset)
    extra_train, extra_val = [], []

    if len(extra_indices):
        extra_labels = y[extra_indices]
        for label in np.unique(extra_labels):
            label_indices = extra_indices[extra_labels == label]

            if len(label_indices) < 2:
                # Keep singleton classes in training; they cannot be
                # independently represented in validation.
                extra_train.extend(label_indices.tolist())
                continue

            tr, va = train_test_split(
                label_indices,
                test_size=0.2,
                random_state=RANDOM_STATE,
                shuffle=True,
            )
            extra_train.extend(tr.tolist())
            extra_val.extend(va.tolist())

    train_idx = np.concatenate([
        np.flatnonzero(train_mask),
        np.asarray(extra_train, dtype=int),
    ])
    val_idx = np.concatenate([
        np.flatnonzero(val_mask),
        np.asarray(extra_val, dtype=int),
    ])
    test_idx = np.flatnonzero(test_mask)

    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError(
            "A split is empty. Check frame values and data sources."
        )

    # Every class in validation must have examples in training.
    missing_train_classes = set(y[val_idx]) - set(y[train_idx])
    if missing_train_classes:
        raise ValueError(
            "Validation contains classes absent from training: "
            f"{sorted(missing_train_classes)}"
        )

    return train_idx, val_idx, test_idx


def main():
    X, y, source, frame, paths, kmeans = load_data()
    train_idx, val_idx, test_idx = make_splits(y, source, frame)

    print(f"Total samples: {len(y)}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Final dataset test samples: {len(test_idx)}")
    print("Classes in complete feature file:", sorted(np.unique(y)))

    if "not_a_sign" not in y:
        print(
            "\nWARNING: not_a_sign is not in this feature file. "
            "This is only a preliminary alphabet model, not the "
            "required final 27-class model."
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Tune on validation only. The final test set is not used here.
    candidates = [
        (1, "scale"),
        (10, "scale"),
        (100, "scale"),
        (1, 0.01),
        (10, 0.01),
        (100, 0.01),
        (1, 0.1),
        (10, 0.1),
        (100, 0.1),
    ]

    best = None
    print("\nTuning SVM hyperparameters...")
    for C, gamma in candidates:
        candidate = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=False,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        candidate.fit(X_train, y_train)
        predictions = candidate.predict(X_val)
        score = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        print(f"C={C:<3} gamma={gamma!s:<6} validation macro-F1={score:.4f}")

        if best is None or score > best["score"]:
            best = {"C": C, "gamma": gamma, "score": score}

    print(f"\nSelected parameters: C={best['C']}, gamma={best['gamma']}")
    print(f"Best validation macro-F1: {best['score']:.4f}")

    # Once parameters are chosen, refit using both training and
    # validation data. Keep the final test block untouched.
    final_idx = np.concatenate([train_idx, val_idx])
    svm = SVC(
        kernel="rbf",
        C=best["C"],
        gamma=best["gamma"],
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    svm.fit(X[final_idx], y[final_idx])

    test_predictions = svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_macro_f1 = f1_score(
        y_test, test_predictions, average="macro", zero_division=0
    )

    print("\nFinal held-out dataset block results")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro-F1: {test_macro_f1:.4f}")
    print(
        classification_report(
            y_test, test_predictions, zero_division=0
        )
    )

    # Required shared model format: dictionary with these exact keys.
    classes = np.unique(y[final_idx])
    joblib.dump(
        {"kmeans": kmeans, "svm": svm, "classes": classes},
        MODEL_PATH,
    )

    results = {
        "samples_total": int(len(y)),
        "samples_train_before_validation_refit": int(len(train_idx)),
        "samples_validation": int(len(val_idx)),
        "samples_test": int(len(test_idx)),
        "classes": classes.tolist(),
        "best_C": best["C"],
        "best_gamma": best["gamma"],
        "validation_macro_f1": float(best["score"]),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_source": "dataset frames >= 800",
        "not_a_sign_present": bool("not_a_sign" in y),
        "test_classification_report": classification_report(
            y_test, test_predictions, output_dict=True,
            zero_division=0,
        ),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {RESULTS_PATH}")
    print("Model dictionary keys: kmeans, svm, classes")
    print("SVM probability support:", svm.probability)


if __name__ == "__main__":
    main()



from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "features_sift.npz"
KMEANS_PATH = ROOT / "models" / "sift_kmeans.joblib"
MODEL_PATH = ROOT / "models" / "sift_svm.joblib"
RESULTS_PATH = ROOT / "models" / "sift_svm_results.json"
RANDOM_STATE = 42


def main():
    data = np.load(FEATURES_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(str)
    source = data["source"].astype(str)
    frame = data["frame"].astype(int)

    kmeans = joblib.load(KMEANS_PATH)

    if X.ndim != 2 or X.shape[1] != kmeans.n_clusters:
        raise ValueError("Feature dimensions do not match the saved KMeans vocabulary.")

    if not np.isfinite(X).all():
        raise ValueError("Feature data contains NaN or infinite values.")

    dataset = source == "dataset"
    train_mask = dataset & (frame < 600)
    val_mask = dataset & (frame >= 600) & (frame < 750)
    test_mask = dataset & (frame >= 800)

    extra_indices = np.flatnonzero(~dataset)
    extra_train = []
    extra_val = []

    for label in np.unique(y[extra_indices]):
        indices = extra_indices[y[extra_indices] == label]

        if len(indices) < 2:
            extra_train.extend(indices.tolist())
        else:
            tr, va = train_test_split(
                indices,
                test_size=0.2,
                random_state=RANDOM_STATE,
                shuffle=True,
            )
            extra_train.extend(tr.tolist())
            extra_val.extend(va.tolist())

    train_idx = np.concatenate([
        np.flatnonzero(train_mask),
        np.asarray(extra_train, dtype=int),
    ])
    val_idx = np.concatenate([
        np.flatnonzero(val_mask),
        np.asarray(extra_val, dtype=int),
    ])
    test_idx = np.flatnonzero(test_mask)

    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError(
            "Training, validation, or test split is empty. "
            "Check data/features_sift.npz."
        )

    missing = set(y[val_idx]) - set(y[train_idx])
    if missing:
        raise ValueError(f"Validation classes missing from training: {sorted(missing)}")

    print("Total samples:", len(y))
    print("Training samples:", len(train_idx))
    print("Validation samples:", len(val_idx))
    print("Held-out dataset test samples:", len(test_idx))
    print("Classes:", sorted(np.unique(y)))

    if "not_a_sign" not in y:
        print(
            "WARNING: not_a_sign is absent. This is a preliminary alphabet "
            "model, not the required final 27-class model."
        )

    candidates = [
        (1, "scale"),
        (10, "scale"),
        (100, "scale"),
        (1, 0.01),
        (10, 0.01),
        (100, 0.01),
        (1, 0.1),
        (10, 0.1),
        (100, 0.1),
    ]

    best = None
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print("\nTuning SVM using validation macro-F1...")
    for C, gamma in candidates:
        model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=False,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        print(f"C={C}, gamma={gamma}: validation macro-F1={score:.4f}")

        if best is None or score > best["validation_macro_f1"]:
            best = {
                "C": C,
                "gamma": gamma,
                "validation_macro_f1": float(score),
            }

    print("\nSelected parameters:", best)

    # Refit using training plus validation data; test data remains held out.
    final_idx = np.concatenate([train_idx, val_idx])
    svm = SVC(
        kernel="rbf",
        C=best["C"],
        gamma=best["gamma"],
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    svm.fit(X[final_idx], y[final_idx])

    X_test, y_test = X[test_idx], y[test_idx]
    predictions = svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    test_macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )
    report = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    print("\nHeld-out dataset block results")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro-F1: {test_macro_f1:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))

    classes = np.unique(y[final_idx])
    joblib.dump(
        {"kmeans": kmeans, "svm": svm, "classes": classes},
        MODEL_PATH,
    )

    results = {
        "samples_total": int(len(y)),
        "samples_train": int(len(train_idx)),
        "samples_validation": int(len(val_idx)),
        "samples_test": int(len(test_idx)),
        "classes": classes.tolist(),
        "best_C": best["C"],
        "best_gamma": best["gamma"],
        "validation_macro_f1": best["validation_macro_f1"],
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(test_macro_f1),
        "test_source": "dataset frames >= 800",
        "not_a_sign_present": bool("not_a_sign" in y),
        "test_classification_report": report,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nSaved model:", MODEL_PATH)
    print("Saved metrics:", RESULTS_PATH)
    print("SVM probability support:", svm.probability)


if __name__ == "__main__":
    main()
