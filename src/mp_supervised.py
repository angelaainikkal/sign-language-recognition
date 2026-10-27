import sys
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)


def load(path):
    d = np.load(path, allow_pickle=True)
    print("Keys in file:", list(d.keys()))
    if "X" in d and "L" in d:            # landmarks.npz: X=features, L=labels
        X, y = d["X"], d["L"]
    elif "L" in d and "y" in d:          # original plan format
        X, y = d["L"], d["y"]
    else:                                # fallback (e.g. SIFT file)
        ykey = next(k for k in ("y", "labels", "label") if k in d)
        xkey = next(k for k in d.keys() if k != ykey)
        X, y = d[xkey], d[ykey]
    return X.reshape(len(X), -1), np.array([str(v) for v in y])


def evaluate(name, model, X, y, cv, report=False):
    p = cross_val_predict(model, X, y, cv=cv)
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro", zero_division=0)
    print(f"{name:<28} accuracy={acc:.4f}  macro-F1={f1:.4f}")
    if report:
        print(classification_report(y, p, zero_division=0))
        labels = sorted(set(y))
        cm = confusion_matrix(y, p, labels=labels)
        pairs = [(cm[i, j], labels[i], labels[j])
                 for i in range(len(labels)) for j in range(len(labels))
                 if i != j and cm[i, j] > 0]
        print("Top confusions (true -> predicted : count):")
        for n, t, q in sorted(pairs, reverse=True)[:8]:
            print(f"  {t} -> {q} : {n}")
    return acc, f1


def main(path):
    X, y = load(path)
    counts = Counter(y)
    print("X:", X.shape, "classes:", len(counts))
    print("Per-class counts:", dict(sorted(counts.items())))

    small = [c for c, n in counts.items() if n < 5]
    if small:
        print("WARNING: very few samples for", sorted(small),
              "- their scores are unreliable")

    keep = np.array([counts[c] >= 2 for c in y])
    X, y = X[keep], y[keep]

    n_splits = min(5, min(Counter(y).values()))
    print("CV folds:", n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\n--- SVM (RBF, balanced) with different C ---")
    best = None
    for C in (1, 10, 100):
        m = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", C=C, gamma="scale",
                              class_weight="balanced"))
        acc, f1 = evaluate(f"SVM C={C}", m, X, y, cv)
        if best is None or f1 > best[0]:
            best = (f1, C)

    C = best[1]
    print(f"\n--- Best SVM: C={C} (detailed report) ---")
    svm = make_pipeline(StandardScaler(),
                        SVC(kernel="rbf", C=C, gamma="scale",
                            class_weight="balanced"))
    evaluate(f"SVM C={C}", svm, X, y, cv, report=True)

    print("--- KNN ---")
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
    evaluate("KNN (k=3)", knn, X, y, cv)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/landmarks.npz")