#!/usr/bin/env python3
# train_xgboost_gpu_safe.py
# XGBoost (ưu tiên GPU) + RandomizedSearchCV hạn chế CPU + Calibration prefit
# Tối ưu để tránh CPU 100% khi train.

import os
# HẠN CHẾ SỐ LUỒNG BLAS/NUMPY (đặt trước khi import numpy/scikit)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
from scipy.stats import randint, uniform
from packaging.version import Version
import xgboost as xgb


# ------------------------- tiện ích -------------------------
def pick_numeric_features(df: pd.DataFrame, drop_cols):
    """Lọc cột số, bỏ URL/Label/…; ép float32, fillna."""
    use = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    for c in use.columns:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.select_dtypes(include=[np.number]).fillna(0.0).astype(np.float32)
    return use


def xgb_gpu_params():
    """Trả về dict tham số GPU phù hợp phiên bản XGBoost."""
    ver = Version(xgb.__version__)
    if ver >= Version("2.0.0"):
        return {"device": "cuda", "tree_method": "hist"}  # v2.x
    else:
        return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}  # v1.x


def save_feature_importance(model: xgb.XGBClassifier, feature_names, outdir: str):
    """Lưu importance theo 'gain', 'weight', 'cover'."""
    os.makedirs(outdir, exist_ok=True)
    booster = model.get_booster()

    def to_df(imp_dict):
        # map f0,f1,... -> tên cột
        rows = []
        for k, v in imp_dict.items():
            # k có dạng 'f123'
            try:
                idx = int(k[1:])
                name = feature_names[idx] if 0 <= idx < len(feature_names) else k
            except Exception:
                name = k
            rows.append((name, float(v)))
        df = pd.DataFrame(rows, columns=["feature", "score"]).sort_values("score", ascending=False)
        return df

    for imp_type in ["gain", "weight", "cover"]:
        imp = booster.get_score(importance_type=imp_type)
        df_imp = to_df(imp)
        df_imp.to_csv(os.path.join(outdir, f"feature_importance_{imp_type}.csv"),
                      index=False, encoding="utf-8-sig")


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\KLTN\UrlPhishing\data\feature_matrix.csv",
                    help="CSV có cột URL/Label và các đặc trưng số.")
    ap.add_argument("--outdir", default=r"E:\KLTN\UrlPhishing\src\artifacts_xgb3.0",
                    help="Thư mục lưu mô hình & báo cáo.")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)

    # kiểm soát mức dùng CPU
    ap.add_argument("--n_iter", type=int, default=25, help="số cấu hình thử trong RandomizedSearchCV")
    ap.add_argument("--cv", type=int, default=3, help="K-fold CV trong search")
    ap.add_argument("--n_jobs_search", type=int, default=1, help="số CPU worker cho RandomizedSearchCV (1 để tránh full CPU)")
    ap.add_argument("--search_subsample", type=float, default=1.0,
                    help="Tỉ lệ dữ liệu train dùng cho search (0< s <=1). Ví dụ 0.3 giảm tải nhanh.")

    # calibration
    ap.add_argument("--calibration", choices=["sigmoid", "isotonic", "none"], default="sigmoid",
                    help="Hiệu chỉnh xác suất: sigmoid nhanh hơn isotonic. 'none' để bỏ.")
    ap.add_argument("--calib_size", type=float, default=0.2,
                    help="Tỉ lệ tách từ train để calibrate trong chế độ prefit.")

    # thiết bị
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="auto: dùng GPU nếu có; 'cuda' ép GPU; 'cpu' tắt GPU.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---------- đọc dữ liệu ----------
    df = pd.read_csv(args.data)
    df.columns = [c.strip() for c in df.columns]

    # chuẩn hóa Label
    label_col = None
    for c in df.columns:
        if c.lower() in ("label", "labels", "y", "target"):
            label_col = c
            break
    if label_col is None:
        raise ValueError("Không tìm thấy cột Label trong CSV.")

    if df[label_col].dtype == object:
        df[label_col] = df[label_col].map({"Phishing": 1, "Legitimate": 0}).astype("float").astype("int")

    # loại cột không phải feature
    drop_cols = ["URL", "Label", "label", "FinalURL"]
    X_all = pick_numeric_features(df, drop_cols=drop_cols)
    y_all = df[label_col].astype(int).values

    if X_all.shape[1] == 0:
        raise ValueError("Không có cột số nào để train XGBoost!")

    feat_names = list(X_all.columns)

    # ---------- split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_all.values.astype(np.float32),
        y_all,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_all
    )

    # ---------- chọn tham số GPU/CPU ----------
    gpu_kwargs = {}
    if args.device == "cuda" or (args.device == "auto"):
        try:
            gpu_kwargs = xgb_gpu_params()
        except Exception:
            gpu_kwargs = {}
    # muốn ép CPU thì để device rỗng

    # ---------- model cơ sở ----------
    base_params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
        n_jobs=1,                # tránh ăn nhiều CPU per-fit
        **gpu_kwargs
    )
    model = xgb.XGBClassifier(**base_params)

    # ---------- không scale: cây không cần, tiết kiệm CPU ----------

    # ---------- RandomizedSearchCV (giảm CPU) ----------
    # Nếu muốn giảm tải: sample một phần dữ liệu train cho search
    X_search = X_train
    y_search = y_train
    if 0 < args.search_subsample < 1.0:
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train, train_size=args.search_subsample,
            stratify=y_train, random_state=args.random_state
        )

    param_dist = {
        "n_estimators": randint(200, 600),
        "max_depth": randint(4, 12),
        "learning_rate": uniform(0.03, 0.27),   # 0.03..0.30
        "subsample": uniform(0.7, 0.3),         # 0.7..1.0
        "colsample_bytree": uniform(0.5, 0.5),  # 0.5..1.0
        "min_child_weight": randint(1, 10),
        "reg_lambda": uniform(0.0, 2.0),
        "reg_alpha": uniform(0.0, 0.5)
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    rs = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="f1",
        cv=cv,
        verbose=2,
        n_jobs=args.n_jobs_search,   # khóa mức song song để CPU không full
        random_state=args.random_state,
        refit=False                 # tự fit lại để kiểm soát prefit calibration
    )
    rs.fit(X_search, y_search)

    best_params = rs.best_params_
    print("[INFO] Best params from search:", best_params)

    # ---------- Fit final (prefit) ----------
    best_model = xgb.XGBClassifier(**base_params, **best_params)
    # để prefit calibration, ta tách 1 phần train làm set calibrate
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=args.calib_size,
        stratify=y_train, random_state=args.random_state
    )
    best_model.fit(X_fit, y_fit)

    # ---------- Calibration (prefit) ----------
    calibrated = None
    if args.calibration != "none":
        from sklearn.calibration import CalibratedClassifierCV
        calibrated = CalibratedClassifierCV(best_model, method=args.calibration, cv="prefit")
        calibrated.fit(X_cal, y_cal)
        clf = calibrated
    else:
        clf = best_model

    # ---------- Evaluate ----------
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "best_params": best_params,
        "n_features": int(X_all.shape[1]),
        "version_xgboost": xgb.__version__,
        "device_used": ("cuda" if "device" in base_params or base_params.get("tree_method") in ("gpu_hist",) else "cpu"),
        "search_subsample": args.search_subsample,
        "cv": args.cv,
        "n_iter": args.n_iter,
        "n_jobs_search": args.n_jobs_search,
        "calibration": args.calibration,
    }

    # ---------- Threshold tuning ----------
    best_thr, best_f1 = 0.5, metrics["f1"]
    for thr in np.linspace(0.1, 0.9, 81):
        yp = (y_proba >= thr).astype(int)
        f1t = f1_score(y_test, yp)
        if f1t > best_f1:
            best_f1, best_thr = f1t, thr
    metrics["best_threshold"] = float(best_thr)
    metrics["best_f1_at_best_threshold"] = float(best_f1)

    # ---------- Confusion matrices ----------
    cm_default = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_default,
                                  display_labels=["Legitimate", "Phishing"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (thr=0.5)")
    plt.savefig(os.path.join(args.outdir, "confusion_matrix_default.png"))
    plt.close()

    y_pred_best = (y_proba >= metrics["best_threshold"]).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)
    disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best,
                                       display_labels=["Legitimate", "Phishing"])
    disp_best.plot(cmap="Greens", values_format="d")
    plt.title(f"Confusion Matrix (best_thr={metrics['best_threshold']:.2f})")
    plt.savefig(os.path.join(args.outdir, "confusion_matrix_best.png"))
    plt.close()

    # ---------- Lưu artifacts ----------
    # mô hình đã calibrate (nếu có), kèm tham số
    model_path = os.path.join(args.outdir, "xgb_calibrated.joblib" if calibrated is not None else "xgb_model.joblib")
    joblib.dump(clf, model_path)

    # lưu tên cột feature
    with open(os.path.join(args.outdir, "feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(feat_names, f, ensure_ascii=False, indent=2)

    # lưu metrics
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # lưu feature importance (từ bản không calibrate: best_model)
    try:
        save_feature_importance(best_model, feat_names, args.outdir)
    except Exception as e:
        print("[WARN] Không thể lưu feature importance:", e)

    print("Training done. Metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
