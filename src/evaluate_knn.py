"""
evaluate_knn.py  -  Member 3: KNN evaluation, K comparison and experiment log

What it does
  1. Loads features (X, y) from .npy files, or a CSV, or demo data
  2. Makes a stratified train/test split (use the SAME split settings as Member 2)
  3. Trains KNN for each K, records Accuracy, Precision, Recall, F1 (macro)
  4. Saves: results table, confusion matrices, graphs, per-class report,
     most-confused class pairs, misclassified sample indices, best model
  5. Optionally logs every run to MLflow
  6. Optionally evaluates Member 2's saved model on the same test split

Quick test (no real data needed):
    python evaluate_knn.py --demo

Real data, .npy files:
    python evaluate_knn.py --X features.npy --y labels.npy

Real data, CSV (one column holds the label):
    python evaluate_knn.py --csv features.csv --label-col label

Extra options:
    --scale                 apply StandardScaler (recommended for most features)
    --ks 3 5 7 9            K values to compare
    --test-size 0.2 --seed 42
    --metric euclidean --weights uniform
    --model knn_model.pkl   also evaluate Member 2's saved model on the test split
    --mlflow                log runs to MLflow (pip install mlflow)

Install:  pip install numpy pandas scikit-learn matplotlib seaborn joblib
"""
import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--X", help="path to features .npy")
    p.add_argument("--y", help="path to labels .npy")
    p.add_argument("--csv", help="path to features CSV")
    p.add_argument("--label-col", default="label", help="label column in CSV")
    p.add_argument("--demo", action="store_true", help="use dummy data")
    p.add_argument("--model", help="Member 2's saved model (.pkl/.joblib)")
    p.add_argument("--ks", type=int, nargs="+", default=[3, 5, 7, 9])
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--weights", default="uniform", choices=["uniform", "distance"])
    p.add_argument("--scale", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--mlflow", action="store_true")
    return p.parse_args()


def load_data(args):
    if args.demo:
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=3000, n_features=64, n_informative=40, n_classes=24,
            n_clusters_per_class=1, class_sep=2.0, random_state=0)
        y = np.array([chr(65 + i) for i in y])
        return X, y
    if args.csv:
        df = pd.read_csv(args.csv)
        return df.drop(columns=[args.label_col]).values, df[args.label_col].values
    if args.X and args.y:
        X = np.load(args.X, allow_pickle=True)
        y = np.load(args.y, allow_pickle=True)
        if X.ndim > 2:  # raw images -> flatten
            X = X.reshape(len(X), -1)
        return X, y
    raise SystemExit("Provide --demo, or --csv, or both --X and --y")


def plot_confusion(cm, labels, title, path):
    size = max(8, len(labels) * 0.45)
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=len(labels) <= 15, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def top_confused(cm, labels, n=5):
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i, j])))
    pairs.sort(key=lambda t: -t[2])
    return pd.DataFrame(pairs[:n], columns=["Actual", "Predicted_as", "Count"])


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    cm_dir = os.path.join(args.out, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    X, y = load_data(args)
    idx = np.arange(len(y))
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes")

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, idx, test_size=args.test_size, random_state=args.seed,
        stratify=y)
    labels = sorted(np.unique(y).tolist())

    if args.mlflow:
        import mlflow
        mlflow.set_experiment("sign-language-knn")

    rows, store = [], {}
    for n, k in enumerate(args.ks, 1):
        clf = KNeighborsClassifier(n_neighbors=k, metric=args.metric,
                                   weights=args.weights)
        model = make_pipeline(StandardScaler(), clf) if args.scale else clf
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        acc = accuracy_score(y_te, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_te, pred, labels=labels)
        cm_path = os.path.join(cm_dir, f"confusion_matrix_k{k}.png")
        plot_confusion(cm, labels, f"Confusion Matrix (K={k})", cm_path)

        params = {"k": k, "metric": args.metric, "weights": args.weights,
                  "scaling": args.scale, "test_size": args.test_size,
                  "seed": args.seed}
        metrics = {"accuracy": acc, "precision_macro": prec,
                   "recall_macro": rec, "f1_macro": f1}
        rows.append({"Experiment": n, "K": k,
                     "Parameters": (f"metric={args.metric}, "
                                    f"weights={args.weights}, scale={args.scale}"),
                     "Accuracy": round(acc, 4), "Precision": round(prec, 4),
                     "Recall": round(rec, 4), "F1": round(f1, 4),
                     "Remarks": ""})
        store[k] = (model, pred, cm)
        print(f"Exp {n}: K={k}  acc={acc:.4f}  P={prec:.4f}  "
              f"R={rec:.4f}  F1={f1:.4f}")

        if args.mlflow:
            with mlflow.start_run(run_name=f"K={k}"):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(cm_path)

    # ---- results table ----
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "experiment_results.csv"), index=False)

    # ---- best K (highest macro F1, then accuracy) ----
    best = res.sort_values(["F1", "Accuracy"], ascending=False).iloc[0]
    best_k = int(best["K"])
    model, pred, cm = store[best_k]
    print(f"\nBest K = {best_k} (F1={best['F1']}, Accuracy={best['Accuracy']})")

    # ---- graphs: K vs metrics ----
    plt.figure(figsize=(6, 4))
    plt.plot(res["K"], res["Accuracy"], marker="o", label="Accuracy")
    plt.plot(res["K"], res["F1"], marker="s", label="F1 (macro)")
    plt.xticks(res["K"])
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.title("K vs Accuracy and F1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "k_vs_metrics.png"), dpi=150)
    plt.close()

    # ---- per-class report for best K ----
    report = classification_report(y_te, pred, labels=labels,
                                   output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.loc[labels]
    per_class.round(4).to_csv(os.path.join(args.out, "per_class_report_bestK.csv"))
    plt.figure(figsize=(max(8, len(labels) * 0.4), 4))
    plt.bar(labels, per_class["f1-score"])
    plt.xlabel("Class")
    plt.ylabel("F1-score")
    plt.title(f"Per-class F1 (K={best_k})")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "per_class_f1_bestK.png"), dpi=150)
    plt.close()

    # ---- error analysis ----
    conf = top_confused(cm, labels)
    conf.to_csv(os.path.join(args.out, "most_confused_pairs.csv"), index=False)
    wrong = pd.DataFrame({"test_sample_index": idx_te[pred != y_te],
                          "actual": y_te[pred != y_te],
                          "predicted": pred[pred != y_te]})
    wrong.to_csv(os.path.join(args.out, "misclassified_samples.csv"), index=False)
    print("\nMost confused pairs:")
    print(conf.to_string(index=False))
    print(f"Misclassified: {len(wrong)} of {len(y_te)} test samples")

    # ---- save best model ----
    joblib.dump(model, os.path.join(args.out, f"best_knn_k{best_k}.pkl"))

    # ---- optional: Member 2's model on the same split ----
    if args.model:
        m2 = joblib.load(args.model)
        p2 = m2.predict(X_te)
        a2 = accuracy_score(y_te, p2)
        _, _, f2, _ = precision_recall_fscore_support(
            y_te, p2, average="macro", zero_division=0)
        print(f"\nMember 2's model on this test split: acc={a2:.4f}, F1={f2:.4f}")
        print("(Only comparable if the split settings match Member 2's.)")

    print(f"\nAll outputs saved in: {args.out}/")


if __name__ == "__main__":
    main()
