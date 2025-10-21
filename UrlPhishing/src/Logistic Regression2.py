#!/usr/bin/env python3
# train_lr_text_plus_numeric.py
# Logistic Regression học đồng thời: TF-IDF(char-ngrams) từ URL + feature số (đã extract)
# - GridSearch trên subsample (vd 30%), rồi refit trên 100% train
# - Calibration (sigmoid) prefit
# - Lưu model, vectorizer, scaler, danh sách feature, metrics, confusion matrices, top coefficients

import os
# Giới hạn luồng BLAS/OpenMP trước khi import numpy/sklearn để tránh full CPU
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "10")

import json
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)


# ---------- tiện ích ----------
def find_col(df: pd.DataFrame, candidates):
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None

def pick_numeric_features(df: pd.DataFrame, drop_cols):
    use = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    for c in use.columns:
        use[c] = pd.to_numeric(use[c], errors="coerce")
    use = use.select_dtypes(include=[np.number]).fillna(0.0).astype(np.float32)
    return use

def build_vectorizer(ngram_min, ngram_max, max_features, min_df, binary, lowercase):
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(ngram_min, ngram_max),
        max_features=max_features,
        min_df=min_df,
        binary=binary,
        lowercase=lowercase,
        dtype=np.float32,
        sublinear_tf=True
    )

def fit_search_lr(X_train_sparse, y_train,
                  C_grid, cv_folds, search_subsample, random_state, n_jobs_search):
    # Subsample để search nhanh hơn
    if 0 < search_subsample < 1.0:
        X_search, _, y_search, _ = train_test_split(
            X_train_sparse, y_train, train_size=search_subsample,
            stratify=y_train, random_state=random_state
        )
    else:
        X_search, y_search = X_train_sparse, y_train

    lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=400,
        class_weight="balanced",
        random_state=random_state,
        tol=1e-4
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        lr,
        param_grid={"C": C_grid},
        scoring="f1",
        cv=cv,
        n_jobs=n_jobs_search,  # 1..2 để không đốt CPU
        verbose=1,
        refit=False
    )
    gs.fit(X_search, y_search)
    print("[INFO] GridSearch best params:", gs.best_params_, "best_score:", gs.best_score_)
    return float(gs.best_params_["C"])

def top_lr_features_from_combined(clf, tfidf, num_feature_names, top_k=50):
    """
    Lấy top hệ số của LR trên ma trận kết hợp [TFIDF | NUM].
    clf: LogisticRegression hoặc CalibratedClassifierCV(prefit LR)
    Trả về 3 DataFrame: dương (class 1), âm (class 0), |coef| lớn nhất.
    """
    lr = None
    if isinstance(clf, LogisticRegression):
        lr = clf
    elif isinstance(clf, CalibratedClassifierCV):
        base = getattr(clf, "base_estimator", None)
        if base is not None and hasattr(base, "coef_"):
            lr = base
        else:
            try:
                lr = clf.calibrated_classifiers_[0].estimator
            except Exception:
                lr = None

    if lr is None or not hasattr(lr, "coef_"):
        return None, None, None

    coefs = lr.coef_.ravel()
    tfidf_names = [f"tfidf:{t}" for t in tfidf.get_feature_names_out()]
    num_names = [f"num:{n}" for n in num_feature_names]
    all_names = np.array(tfidf_names + num_names)

    df = pd.DataFrame({"feature": all_names, "coef": coefs})
    df_pos = df.sort_values("coef", ascending=False).head(top_k).reset_index(drop=True)
    df_neg = df.sort_values("coef", ascending=True).head(top_k).reset_index(drop=True)
    df_abs = df.assign(abs_coef=np.abs(df["coef"])).sort_values("abs_coef", ascending=False).head(top_k).reset_index(drop=True)
    return df_pos, df_neg, df_abs


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # IO
    ap.add_argument("--data", default=r"E:\KLTN\UrlPhishing\data\feature_matrix.csv",
                    help="CSV có cột URL + Label + feature số đã extract.")
    ap.add_argument("--outdir", default=r"E:\KLTN\UrlPhishing\src\artifacts_lr2",
                    help="Thư mục lưu model & báo cáo.")

    # Vectorizer
    ap.add_argument("--ngram_min", type=int, default=3)
    ap.add_argument("--ngram_max", type=int, default=4)
    ap.add_argument("--max_features", type=int, default=30000)
    ap.add_argument("--min_df", type=int, default=3)
    ap.add_argument("--binary", action="store_true", help="Dùng TF nhị phân thay vì TF-IDF")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase URL trước khi vector hoá (mặc định False)")

    # Train / Search
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--search_subsample", type=float, default=0.3, help="Tỉ lệ train dùng cho GridSearch (0<s<=1)")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--n_jobs_search", type=int, default=1)
    ap.add_argument("--C_grid", type=str, default="0.1,0.3,1,3,10",
                    help="Danh sách C (phẩy) cho GridSearch, vd: 0.1,0.3,1,3,10")

    # Calibration
    ap.add_argument("--calibration", choices=["sigmoid", "isotonic", "none"], default="sigmoid")
    ap.add_argument("--calib_size", type=float, default=0.2, help="Tỉ lệ train dành cho calibration (prefit)")

    # Reporting
    ap.add_argument("--top_k", type=int, default=50, help="Số feature top để lưu/hiển thị")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ----- Load dữ liệu -----
    df = pd.read_csv(args.data)
    df.columns = [c.strip() for c in df.columns]

    url_col = find_col(df, {"url", "link", "uri", "address"})
    label_col = find_col(df, {"label", "labels", "y", "target"})
    if url_col is None or label_col is None:
        raise ValueError("CSV phải có cột URL và Label.")

    y = df[label_col]
    if y.dtype == object:
        y = y.map({"Phishing": 1, "Legitimate": 0})
    y = y.astype(int).values

    X_text_all = df[url_col].astype(str).values
    # numeric features = tất cả cột số ngoại trừ URL/Label và các cột bắt đầu bằng '_'
    drop_cols = [url_col, label_col]
    X_num_df = pick_numeric_features(df, drop_cols=drop_cols + [c for c in df.columns if c.startswith("_")])
    num_feature_names = list(X_num_df.columns)

    # ----- Split -----
    X_txt_train, X_txt_test, y_train, y_test, X_num_train_df, X_num_test_df = train_test_split(
        X_text_all, y, X_num_df, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # ----- Vector hoá URL -----
    tfidf = build_vectorizer(
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_features=args.max_features,
        min_df=args.min_df,
        binary=args.binary,
        lowercase=args.lowercase
    )
    X_text_train = tfidf.fit_transform(X_txt_train)
    X_text_test  = tfidf.transform(X_txt_test)

    # ----- Scale numeric & ghép ma trận -----
    # StandardScaler(with_mean=False) để giữ dạng sparse
    scaler = StandardScaler(with_mean=False)
    X_num_train = csr_matrix(X_num_train_df.values.astype(np.float32))
    X_num_test  = csr_matrix(X_num_test_df.values.astype(np.float32))

    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled  = scaler.transform(X_num_test)

    # Ghép [TFIDF | NUM]
    X_train = hstack([X_text_train, X_num_train_scaled]).tocsr()
    X_test  = hstack([X_text_test,  X_num_test_scaled]).tocsr()

    # ----- GridSearch trên subsample -----
    C_grid = [float(x.strip()) for x in args.C_grid.split(",") if x.strip()]
    best_C = fit_search_lr(
        X_train, y_train,
        C_grid=C_grid, cv_folds=args.cv,
        search_subsample=args.search_subsample,
        random_state=args.random_state,
        n_jobs_search=args.n_jobs_search
    )

    # ----- Prefit calibration -----
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=args.calib_size, stratify=y_train, random_state=args.random_state
    )

    base_lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=best_C,
        max_iter=500,
        class_weight="balanced",
        random_state=args.random_state,
        tol=1e-4
    )
    base_lr.fit(X_fit, y_fit)

    if args.calibration != "none":
        calib = CalibratedClassifierCV(base_lr, method=args.calibration, cv="prefit")
        calib.fit(X_cal, y_cal)
        clf = calib
    else:
        clf = base_lr

    # ----- Evaluate -----
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "best_C": best_C,
        "vectorizer": {
            "analyzer": "char_wb",
            "ngram_range": [args.ngram_min, args.ngram_max],
            "max_features": args.max_features,
            "min_df": args.min_df,
            "binary": bool(args.binary),
            "lowercase": bool(args.lowercase)
        },
        "n_num_features": len(num_feature_names),
        "cv": args.cv,
        "search_subsample": args.search_subsample,
        "calibration": args.calibration
    }

    # Tune threshold (maximize F1)
    best_thr, best_f1 = 0.5, metrics["f1"]
    for thr in np.linspace(0.1, 0.9, 81):
        yp = (y_proba >= thr).astype(int)
        f1t = f1_score(y_test, yp)
        if f1t > best_f1:
            best_f1, best_thr = f1t, thr
    metrics["best_threshold"] = float(best_thr)
    metrics["best_f1_at_best_threshold"] = float(best_f1)

    # ----- Confusion matrices -----
    cm_default = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_default,
                                  display_labels=["Legitimate", "Phishing"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("LR (TFIDF + NUM) — Confusion Matrix (thr=0.5)")
    plt.savefig(os.path.join(args.outdir, "confusion_matrix_default.png"))
    plt.close()

    y_pred_best = (y_proba >= metrics["best_threshold"]).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)
    disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best,
                                       display_labels=["Legitimate", "Phishing"])
    disp_best.plot(cmap="Greens", values_format="d")
    plt.title(f"LR (TFIDF + NUM) — Confusion Matrix (best_thr={metrics['best_threshold']:.2f})")
    plt.savefig(os.path.join(args.outdir, "confusion_matrix_best.png"))
    plt.close()

    # ----- Lưu artifacts -----
    joblib.dump(clf, os.path.join(args.outdir, "logreg_text_num_calibrated.joblib" if args.calibration!="none" else "logreg_text_num.joblib"))
    joblib.dump(tfidf, os.path.join(args.outdir, "tfidf_vectorizer.joblib"))
    joblib.dump(scaler, os.path.join(args.outdir, "scaler_num.joblib"))
    with open(os.path.join(args.outdir, "num_feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(num_feature_names, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ----- Top coefficients (kèm prefix nguồn) -----
    df_pos, df_neg, df_abs = top_lr_features_from_combined(clf, tfidf, num_feature_names, top_k=args.top_k)
    if df_pos is not None:
        df_pos.to_csv(os.path.join(args.outdir, "top_pos_coef.csv"), index=False, encoding="utf-8-sig")
        df_neg.to_csv(os.path.join(args.outdir, "top_neg_coef.csv"), index=False, encoding="utf-8-sig")
        df_abs.to_csv(os.path.join(args.outdir, "top_abs_coef.csv"), index=False, encoding="utf-8-sig")

        print("\n[Top + coef → class 1 (phishing)]")
        print(df_pos.head(10).to_string(index=False))
        print("\n[Top - coef → class 0 (legitimate)]")
        print(df_neg.head(10).to_string(index=False))
        print("\n[Top |coef|]")
        print(df_abs.head(10).to_string(index=False))
    else:
        print("[WARN] Không trích được hệ số LR (coef_) để giải thích.")

    print("\nTraining done. Metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
