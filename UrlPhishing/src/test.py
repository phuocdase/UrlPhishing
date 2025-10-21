# src/eval_blend_fixed.py
# Evaluate blended model (LR TF-IDF + XGB numeric) on a fixed test file/folder.
# - Nếu TEST_PATH là file .csv -> chấm 1 file
# - Nếu TEST_PATH là thư mục   -> chấm tất cả *.csv trong thư mục

import os, sys, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# =========================
#        FIXED PATHS
# =========================
MODEL_PATH     = r"E:\KLTN\UrlPhishing\src\artifacts_blend\blend_model.joblib"
TEST_PATH      = r"C:\Users\damlo\Downloads\URL dataset.csv"
OUTDIR_BASE    = r"E:\KLTN\UrlPhishing\src\artifacts_eval"
URL_COL_NAME   = "URL"
LABEL_COL_NAME = "Label"

# Cách tính metrics:
USE_MODEL_THRESHOLD = True
FIND_BEST_THR       = True


# =========================
#    Unpickle wrapper
# =========================
class BlendLRXGB:
    def __init__(self, lr, vec, xgb, scaler, num_cols, text_col="URL", alpha=0.8, threshold=0.37):
        self.lr = lr
        self.vec = vec
        self.xgb = xgb
        self.scaler = scaler
        self.num_cols = list(num_cols)
        self.text_col = text_col
        self.alpha = float(alpha)
        self.threshold = float(threshold)

    @staticmethod
    def _clean_numeric(df, cols):
        X = df[cols].copy().apply(pd.to_numeric, errors="coerce").astype(np.float32)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        med = X.median(numeric_only=True)
        X = X.fillna(med).fillna(0.0)
        if not np.isfinite(X.to_numpy()).all():
            X = pd.DataFrame(
                np.nan_to_num(X.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
                columns=X.columns, dtype=np.float32
            )
        return X.values.astype(np.float32)

    def predict_proba_from_df(self, df: pd.DataFrame):
        # LR (TF-IDF)
        if isinstance(self.lr, Pipeline):
            p_lr = self.lr.predict_proba(df[self.text_col].astype(str).values)[:, 1]
        else:
            assert self.vec is not None, "Blend model needs vectorizer for LR."
            Xt = self.vec.transform(df[self.text_col].astype(str).values)
            if hasattr(self.lr, "predict_proba"):
                p_lr = self.lr.predict_proba(Xt)[:, 1]
            else:
                s = self.lr.decision_function(Xt); p_lr = 1/(1+np.exp(-s))

        # XGB (numeric)
        Xn = self._clean_numeric(df, self.num_cols) if self.num_cols else np.zeros((len(df),0), np.float32)
        if self.scaler is not None and Xn.size > 0:
            Xn = self.scaler.transform(Xn)
        if hasattr(self.xgb, "predict_proba"):
            p_xgb = self.xgb.predict_proba(Xn)[:, 1]
        elif hasattr(self.xgb, "decision_function"):
            s = self.xgb.decision_function(Xn); p_xgb = 1/(1+np.exp(-s))
        else:
            p_xgb = self.xgb.predict(Xn).astype(float)

        return self.alpha * p_lr + (1.0 - self.alpha) * p_xgb

    def predict_from_df(self, df: pd.DataFrame):
        return (self.predict_proba_from_df(df) >= self.threshold).astype(int)


# =========================
#   Import extractor
# =========================
def add_data_dir_to_syspath(me: Path):
    candidates = [
        me / "data",
        me.parent / "data",
        Path(r"E:\KLTN\UrlPhishing\data"),
    ]
    for d in candidates:
        if (d / "extract.py").exists():
            sys.path.insert(0, str(d))
            return str(d)
    return None

THIS_DIR = Path(__file__).resolve().parent
_ = add_data_dir_to_syspath(THIS_DIR)

try:
    from extract import extraction as extract_record
except Exception:
    extract_record = None  # nếu CSV đã có đủ numeric cols thì không cần


# =========================
#     Helper functions
# =========================
def build_input_df(df_raw: pd.DataFrame, model: BlendLRXGB, url_col="URL"):
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # tìm cột URL
    url_col_found = next((c for c in [url_col, "URL", "url", "Url"] if c in df.columns), None)
    if url_col_found is None:
        raise ValueError("CSV cần có cột URL/url/Url.")

    # base: cột URL cho LR
    out = pd.DataFrame(index=df.index)
    out[model.text_col] = df[url_col_found].astype(str)

    have = [c for c in model.num_cols if c in df.columns]
    miss = [c for c in model.num_cols if c not in df.columns]

    if not miss:
        # đủ numeric -> dùng thẳng
        out = pd.concat([out, df[model.num_cols]], axis=1)
        return out

    if extract_record is None:
        raise RuntimeError(
            f"Missing numeric columns {miss[:8]}... and cannot import data/extract.py. "
            f"Either add those columns to CSV or ensure data/extract.py is importable."
        )

    # bù numeric bằng extractor
    feats_list = []
    for u in df[url_col_found].astype(str).tolist():
        feats = extract_record(u, label=None) or {}
        feats_list.append(feats)
    feats_df = pd.DataFrame(feats_list)

    for c in model.num_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = pd.to_numeric(feats_df.get(c, 0.0), errors="coerce")

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(0.0, inplace=True)
    return out


def metrics_at_threshold(y_true, y_proba, thr):
    y_hat = (y_proba >= thr).astype(int)
    m = {
        "threshold": float(thr),
        "f1": float(f1_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat)),
        "recall": float(recall_score(y_true, y_hat)),
    }
    try:
        m["auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception as e:
        m["auc"] = None
        m["auc_error"] = f"{type(e).__name__}: {e}"
    return m


def save_confmat(y_true, y_hat, outdir, title):
    cm = confusion_matrix(y_true, y_hat)
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Phishing"]).plot(
        cmap="Blues", values_format="d"
    )
    plt.title(title)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(outdir, "confusion_matrix.png"), dpi=160, bbox_inches="tight")
    plt.close()


def find_best_threshold(y_true, y_proba):
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.05, 0.95, 91):
        f1t = f1_score(y_true, (y_proba >= thr).astype(int))
        if f1t > best_f1:
            best_f1, best_thr = f1t, thr
    return float(best_thr), float(best_f1)


def eval_one_csv(csv_path: Path, model: BlendLRXGB, outdir_base: Path):
    outdir = outdir_base / csv_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluating: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # build input
    df_in = build_input_df(df_raw, model, url_col=URL_COL_NAME)

    # predict
    p_blend = model.predict_proba_from_df(df_in)
    thr_use = model.threshold if USE_MODEL_THRESHOLD else 0.5
    y_hat = (p_blend >= thr_use).astype(int)

    # save predictions
    pred_df = df_raw.copy()
    pred_df["p_blend"] = p_blend
    pred_df["pred_blend"] = y_hat
    pred_df.to_csv(outdir / "predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "csv": str(csv_path),
        "num_rows": int(len(df_raw)),
        "used_threshold": float(thr_use),
        "model_threshold": float(model.threshold),
        "alpha_in_model": float(model.alpha),
    }

    if LABEL_COL_NAME in df_raw.columns:
        y_true = df_raw[LABEL_COL_NAME]
        if y_true.dtype == object:
            y_true = y_true.map({"Phishing": 1, "Legitimate": 0})
        y_true = y_true.astype(int).values

        # metrics @ thr_use
        m = metrics_at_threshold(y_true, p_blend, thr_use)
        summary["metrics_at_used_threshold"] = m
        save_confmat(y_true, (p_blend >= thr_use).astype(int), outdir, f"{csv_path.name} (thr={thr_use:.2f})")

        # optional: best threshold on this CSV
        if FIND_BEST_THR:
            best_thr, best_f1 = find_best_threshold(y_true, p_blend)
            summary["best_threshold_on_this_csv"] = best_thr
            summary["best_f1_at_best_threshold"] = best_f1

            # save confusion at best thr
            y_best = (p_blend >= best_thr).astype(int)
            cm_dir = outdir / "best_threshold"
            cm_dir.mkdir(exist_ok=True)
            save_confmat(y_true, y_best, cm_dir, f"{csv_path.name} (best_thr={best_thr:.2f})")

    # save per-file summary
    with open(outdir / "metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved -> {outdir}")
    return summary


def main():
    model: BlendLRXGB = joblib.load(MODEL_PATH)
    outdir_base = Path(OUTDIR_BASE)
    outdir_base.mkdir(parents=True, exist_ok=True)

    test_path = Path(TEST_PATH)
    summaries = []

    if test_path.is_file():
        summaries.append(eval_one_csv(test_path, model, outdir_base))
    elif test_path.is_dir():
        csvs = sorted([p for p in test_path.glob("*.csv")])
        if not csvs:
            print(f"[WARN] Không tìm thấy .csv trong thư mục: {test_path}")
            return
        for csv in csvs:
            summaries.append(eval_one_csv(csv, model, outdir_base))
    else:
        print(f"[ERR] TEST_PATH không tồn tại: {test_path}")
        return

    # tổng hợp (nếu có Label trong tất cả file)
    summary_path = outdir_base / "summary_all.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"[DONE] All results saved under: {outdir_base}")


if __name__ == "__main__":
    main()
