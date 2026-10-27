import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def load(path):
    d = np.load(path, allow_pickle=True)
    print("Keys in file:", list(d.keys()))
    # landmarks.npz uses L and y; for the SIFT file we guess the key names
    if "L" in d:
        X, y = d["L"], d["y"]
    else:
        ykey = next(k for k in ("y", "labels", "label") if k in d)
        xkey = next(k for k in d.keys() if k != ykey)
        X, y = d[xkey], d[ykey]
    X = X.reshape(len(X), -1)   # flatten (N,21,3) -> (N,63)
    return X, y

def main(path):
    X, y = load(path)
    print("X:", X.shape, "classes:", len(np.unique(y)))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        print(f"{name}: accuracy={accuracy_score(yte, p):.4f}  "
              f"macro-F1={f1_score(yte, p, average='macro'):.4f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/features_sift.npz")