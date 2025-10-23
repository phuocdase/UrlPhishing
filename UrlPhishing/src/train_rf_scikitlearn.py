%%writefile train_rf.py
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import joblib


def sample_params(rng):
    """Randomly sample hyperparameters within sensible ranges for sklearn RF."""
    n_estimators = int(rng.integers(200, 901))
    max_depth = int(rng.integers(8, 33))
    max_features_choices = ["sqrt", "log2", 1.0, 0.5]
    max_features = max_features_choices[int(rng.integers(0, len(max_features_choices)))]
    bootstrap = bool(rng.integers(0, 2))
    min_samples_split = int(rng.integers(2, 17))
    min_samples_leaf = int(rng.integers(1, 9))

    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "criterion": "gini",
        "random_state": None,
        "n_jobs": -1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/feature-matrix-shortredirect/feature_matrix.csv",
                    help="Đường dẫn CSV có cột URL, Label và các đặc trưng số")
    ap.add_argument("--outdir", default="artifacts_rf_cpu", help="Thư mục lưu mô hình và báo cáo")
    ap.add_argument("--n_iter", type=int, default=30, help="Số cấu hình tham số thử (random search - holdout)")
    ap.add_argument("--test_size", type=float, default=0.2, help="Tỷ lệ test")
    ap.add_argument("--val_size", type=float, default=0.15, help="Tỷ lệ validation từ train")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--no_scale", action="store_true", help="Không scale đặc trưng (cây không cần scale)")
    args = ap.parse_args()


    os.makedirs(args.outdir, exist_ok=True)

    # Đọc dữ liệu
    df = pd.read_csv(args.data)
    df.columns = [c.strip() for c in df.columns]

    # Chuẩn hóa Label về {0,1}
    if df["Label"].dtype == object:
        df["Label"] = df["Label"].map({"Phishing": 1, "Legitimate": 0})

    drop_cols = ["URL", "Label"]
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.startswith("_")]
    if len(feature_cols) == 0:
        raise ValueError("Không tìm thấy cột đặc trưng!")

    X = df[feature_cols].astype(np.float32)
    y = df["Label"].astype(np.int32)

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Scale (tùy chọn)
    if args.no_scale:
        X_train_scaled, X_test_scaled = X_train, X_test
        scaler_info = {"used": False}
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scaler_info = {"used": True, "class": "sklearn.preprocessing.StandardScaler"}

    # Chia validation từ train
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=args.val_size, random_state=args.random_state, stratify=y_train
    )

    rng = np.random.default_rng(args.random_state)
    best_params = None
    best_f1 = -1.0

    for i in range(args.n_iter):
        params = sample_params(rng)
        rf = RandomForestClassifier(**params)
        rf.fit(X_tr, y_tr)
        y_val_pred = rf.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    # Train mô hình cuối cùng trên toàn bộ train (train+val)
    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train_scaled, y_train)

    # Đánh giá trên test
    y_pred = best_rf.predict(X_test_scaled)
    y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "best_params": best_params,
        "scaler": scaler_info,
    }

    # Tối ưu ngưỡng quyết định theo F1
    best_thr, best_f1_test = 0.5, metrics["f1"]
    for thr in np.linspace(0.1, 0.9, 81):
        yp = (y_proba >= thr).astype(int)
        f1t = f1_score(y_test, yp)
        if f1t > best_f1_test:
            best_f1_test, best_thr = f1t, thr
    metrics["best_threshold"] = float(best_thr)
    metrics["best_f1_at_best_threshold"] = float(best_f1_test)

    # Confusion matrix mặc định
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Phishing"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest (CPU) - Confusion Matrix (threshold=0.5)")
    plt.savefig(os.path.join(args.outdir, "confusion_matrix_default.png"))
    plt.close()

    # Confusion matrix ở ngưỡng tối ưu
    y_pred_best = (y_proba >= metrics["best_threshold"]).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)
    disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=["Legitimate", "Phishing"])
    disp_best.plot(cmap="Greens", values_format="d")
    plt.title(f"Random Forest (CPU) - Confusion Matrix (best_threshold={metrics['best_threshold']:.2f})")
    plt.savefig(os.path.join(args.outdir, "confusion_matrix_best.png"))
    plt.close()

    # Feature importances
    importances = getattr(best_rf, "feature_importances_", None)
    if importances is not None:
        fi = (
            pd.DataFrame({"feature": feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
        )
        fi.to_csv(os.path.join(args.outdir, "rf_cpu_feature_importances.csv"), index=False)

    # Lưu model và metadata
    joblib.dump(best_rf, os.path.join(args.outdir, "rf_sklearn.joblib"))
    with open(os.path.join(args.outdir, "feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("✅ Done. Metrics:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
