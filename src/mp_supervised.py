import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load(path):
    d = np.load(path, allow_pickle=True)
    print("Keys in file:", list(d.keys()))
    if "X" in d and "L" in d:
      X, y = d["X"], d["L"]
    else:
        ykey = next(k for k in ("y", "labels", "label") if k in d)
        xkey = next(k for k in d.keys() if k != ykey)
        X, y = d[xkey], d[ykey]
    return X.reshape(len(X), -1), y

def main(path):
    X, y = load(path)
    counts = Counter(y)
    print("X:", X.shape, "classes:", len(counts))
    print("Per-class counts:", dict(sorted(counts.items())))

    # drop classes too small to cross-validate
    keep = np.array([counts[c] >= 2 for c in y])
    if not keep.all():
        print("Dropping classes with <2 samples:",
              [c for c, n in counts.items() if n < 2])
        X, y = X[keep], y[keep]

    n_splits = min(5, min(Counter(y).values()))
    print("CV folds:", n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "SVM (RBF, balanced)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")),
        "KNN (k=3)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
    }
    for name, m in models.items():
        p = cross_val_predict(m, X, y, cv=cv)
        print(f"\n{name}: accuracy={accuracy_score(y, p):.4f}  "
              f"macro-F1={f1_score(y, p, average='macro', zero_division=0):.4f}")
        if "SVM" in name:
            print(classification_report(y, p, zero_division=0))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/landmarks.npz")
