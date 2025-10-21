
"""
Train URL classifier with SVM (Linear, RBF, or SGD-hinge) for phishing detection.

- Cleans data (coerce numeric, handle NaN/Inf).
- Uses StandardScaler.
- Hyperparameter search with RandomizedSearchCV.
- Calibrates probabilities where needed.
- Optimizes decision threshold for F1.
- Saves metrics, confusion matrices, model, and feature order.
"""
import argparse
import json
import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from scipy.stats import loguniform, uniform
import joblib


def resolve_data_path(p: str) -> str:
    if p and os.path.exists(p):
        return p
    candidates = []
    common = [
        "/kaggle/working/feature_matrix_clean.csv",
        "/mnt/data/feature_matrix_clean.csv",
    ]
    common += glob.glob("/kaggle/input/**/feature_matrix_clean.csv", recursive=True)
    for c in common:
        if os.path.exists(c):
            candidates.append(c)
    if candidates:
        print(f"[INFO] Using discovered data at: {candidates[0]} (requested: {p})")
        return candidates[0]
    raise FileNotFoundError("Data CSV not found. Provide --data explicitly.")


def clean_features(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    if not np.isfinite(X.to_numpy()).all():
        X = pd.DataFrame(np.nan_to_num(X.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
                         columns=X.columns, dtype=np.float32)
    return X


def evaluate_and_save(y_test, y_proba, outdir, title_prefix, labels=("Legitimate", "Phishing")):
    os.makedirs(outdir, exist_ok=True)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
    }
    best_thr, best_f1 = 0.5, metrics["f1"]
    for thr in np.linspace(0.1, 0.9, 81):
        yp = (y_proba >= thr).astype(int)
        f1t = f1_score(y_test, yp)
        if f1t > best_f1:
            best_f1, best_thr = f1t, thr
    metrics["best_threshold"] = float(best_thr)
    metrics["best_f1_at_best_threshold"] = float(best_f1)

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=list(labels)).plot(cmap="Blues", values_format="d")
    plt.title(f"{title_prefix} - Confusion Matrix (threshold=0.5)")
    plt.savefig(os.path.join(outdir, "confusion_matrix_default.png"))
    plt.close()

    y_pred_best = (y_proba >= metrics["best_threshold"]).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)
    ConfusionMatrixDisplay(cm_best, display_labels=list(labels)).plot(cmap="Greens", values_format="d")
    plt.title(f"{title_prefix} - Confusion Matrix (best_threshold={metrics['best_threshold']:.2f})")
    plt.savefig(os.path.join(outdir, "confusion_matrix_best.png"))
    plt.close()

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\KLTN\UrlPhishing\data\feature_matrix_clean.csv",
                    help="CSV with URL, Label, and numeric features")
    ap.add_argument("--outdir", default="artifacts_svm", help="Output directory")
    ap.add_argument("--model", choices=["linear", "rbf", "sgd"], default="linear",
                    help="SVM variant: linear (LinearSVC+calibration), rbf (SVC), or sgd (SGD hinge + calibration)")
    ap.add_argument("--n_iter", type=int, default=30, help="RandomizedSearch iterations")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args, unknown = ap.parse_known_args()
    if unknown:
        print("[WARN] Ignoring unknown args:", unknown)

    os.makedirs(args.outdir, exist_ok=True)

    data_path = resolve_data_path(args.data)
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]

    if df["Label"].dtype == object:
        df["Label"] = df["Label"].map({"Phishing": 1, "Legitimate": 0}).astype(int)

    drop_cols = [c for c in ["URL", "Label"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.startswith("_")]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found!")

    X = clean_features(df, feature_cols).values
    y = df["Label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.random_state)

    best_estimator = None
    best_params = None
    title = ""

    if args.model == "linear":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced", random_state=args.random_state))
        ])
        param_dist = {
            "clf__C": loguniform(1e-3, 1e2),
            "clf__loss": ["hinge", "squared_hinge"],
        }
        rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=args.n_iter,
                                scoring="f1", cv=cv, verbose=2, n_jobs=-1, random_state=args.random_state)
        rs.fit(X_train, y_train)
        best_params = rs.best_params_

        calibrated = CalibratedClassifierCV(rs.best_estimator_, cv=3, method="sigmoid")
        calibrated.fit(X_train, y_train)
        best_estimator = calibrated
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
        title = "Linear SVM (LinearSVC + calibration)"

        # Save linear weights
        try:
            lin = rs.best_estimator_.named_steps["clf"]
            coef = lin.coef_.ravel()
            weights = (pd.DataFrame({"feature": feature_cols, "weight": coef})
                       .sort_values("weight", ascending=False))
            weights.to_csv(os.path.join(args.outdir, "linear_svm_feature_weights.csv"), index=False)
        except Exception:
            pass

    elif args.model == "rbf":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True,
                        random_state=args.random_state))
        ])
        param_dist = {
            "clf__C": loguniform(1e-2, 1e2),
            "clf__gamma": loguniform(1e-4, 1e0),
        }
        rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=args.n_iter,
                                scoring="f1", cv=cv, verbose=2, n_jobs=-1, random_state=args.random_state)
        rs.fit(X_train, y_train)
        best_estimator = rs.best_estimator_
        best_params = rs.best_params_
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
        title = "RBF SVM (SVC)"

    else:  # sgd
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(loss="hinge", class_weight="balanced",
                                  random_state=args.random_state, max_iter=2000))
        ])
        param_dist = {
            "clf__alpha": loguniform(1e-6, 1e-2),
            "clf__l1_ratio": uniform(0.0, 1.0),
            "clf__penalty": ["l2", "l1", "elasticnet"],
        }
        rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=args.n_iter,
                                scoring="f1", cv=cv, verbose=2, n_jobs=-1, random_state=args.random_state)
        rs.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(rs.best_estimator_, cv=3, method="sigmoid")
        calibrated.fit(X_train, y_train)
        best_estimator = calibrated
        best_params = rs.best_params_
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
        title = "Linear SVM (SGD hinge + calibration)"

    metrics = evaluate_and_save(y_test, y_proba, args.outdir, title)
    metrics["best_params"] = best_params
    metrics["model"] = args.model

    joblib.dump(best_estimator, os.path.join(args.outdir, "svm_model.joblib"))
    with open(os.path.join(args.outdir, "feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Training done. Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
