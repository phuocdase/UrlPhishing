# src/app.py — Streamlit app cho model blend (LR TF-IDF + XGB numeric)
# Dùng extractor từ:  E:\KLTN\UrlPhishing\data\extract.py

import sys
import json
from pathlib import Path
from urllib.parse import urlparse
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# =============== PATHS ===============
# model blend đã đóng gói
MODEL_PATH = r"E:\KLTN\UrlPhishing\src\artifacts_blend\blend_model.joblib"

# thêm .../data vào sys.path để import extract.py
APP_DIR = Path(__file__).resolve().parent
CANDIDATE_DATA_DIRS = [
    APP_DIR / "data",
    APP_DIR.parent / "data",                 # ../data (trường hợp app chạy từ src/)
    Path(r"E:\KLTN\UrlPhishing\data\extract.py"),       # đường dẫn tuyệt đối (dự án của bạn)
]
for d in CANDIDATE_DATA_DIRS:
    if (d / "extract.py").exists():
        sys.path.insert(0, str(d))
        break

# import extractor trong data/extract.py
try:
    from extract import extraction as extract_record
except Exception as e:
    st.error(f"Không thể import extraction từ data/extract.py: {type(e).__name__}: {e}")
    st.stop()


# =============== RE-DECLARE WRAPPER để unpickle ===============
class BlendLRXGB:
    """Wrapper đã dùng khi pack model; cần khai báo lại để joblib.load hoạt động."""
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
            X = pd.DataFrame(np.nan_to_num(X.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0),
                             columns=X.columns, dtype=np.float32)
        return X.values.astype(np.float32)

    def predict_proba_from_df(self, df: pd.DataFrame):
        # LR: TF-IDF từ cột text_col
        if isinstance(self.lr, Pipeline):
            p_lr = self.lr.predict_proba(df[self.text_col].astype(str).values)[:, 1]
        else:
            assert self.vec is not None, "Blend model cần vectorizer cho LR."
            Xt = self.vec.transform(df[self.text_col].astype(str).values)
            if hasattr(self.lr, "predict_proba"):
                p_lr = self.lr.predict_proba(Xt)[:, 1]
            else:
                s = self.lr.decision_function(Xt); p_lr = 1/(1+np.exp(-s))

        # XGB: numeric theo num_cols (+ scaler nếu có)
        Xn = self._clean_numeric(df, self.num_cols) if self.num_cols else np.zeros((len(df), 0), np.float32)
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


# =============== LOAD MODEL ===============
model = joblib.load(MODEL_PATH)  # hoạt động vì đã có class BlendLRXGB


# =============== HELPERS ===============
def build_row_from_url(url_str: str) -> pd.DataFrame:
    """Gọi extractor trong data/extract.py, sau đó map đúng schema model: URL + numeric cols."""
    feats = extract_record(str(url_str), label=None) or {}
    row = {c: 0.0 for c in model.num_cols}
    for k, v in feats.items():
        if k in row:
            row[k] = v
    row["URL"] = str(url_str)  # cho LR TF-IDF
    return pd.DataFrame([row])


def build_input_from_csv(df_csv: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá CSV người dùng:
       - Yêu cầu phải có cột URL/url/Url;
       - Nếu thiếu bất kỳ numeric col nào trong model.num_cols → tự extract và bù.
    """
    df = df_csv.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # tìm cột URL
    url_col = next((c for c in ["URL", "url", "Url"] if c in df.columns), None)
    if url_col is None:
        raise ValueError("CSV cần có cột 'URL' (hoặc 'url'/'Url').")

    out = pd.DataFrame(index=df.index)
    out["URL"] = df[url_col].astype(str)

    have = [c for c in model.num_cols if c in df.columns]
    miss = [c for c in model.num_cols if c not in df.columns]

    if not miss:
        # đã đủ numeric → dùng thẳng
        out = pd.concat([out, df[model.num_cols]], axis=1)
        return out

    # thiếu → extract từng URL để bù numeric
    st.info(f"CSV thiếu {len(miss)} cột numeric. Đang sinh thêm từ data/extract.py…")
    feats_list = []
    prog = st.progress(0)
    total = len(df)
    for i, u in enumerate(df[url_col].astype(str).tolist(), start=1):
        feats = extract_record(u, label=None) or {}
        feats_list.append(feats)
        if i % 20 == 0 or i == total:
            prog.progress(i / total)
    prog.empty()
    feats_df = pd.DataFrame(feats_list)

    for c in model.num_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = pd.to_numeric(feats_df.get(c, 0.0), errors="coerce")

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(0.0, inplace=True)
    return out


# =============== UI ===============
st.set_page_config(page_title="URL Phishing (Blend + extract.py)", page_icon="", layout="centered")
st.title("URL Phishing Detection — Blend LR (TF-IDF) + XGB (numeric)")

with st.expander("ℹModel info", expanded=True):
    st.write(f"- **Alpha (LR weight):** `{model.alpha}`")
    st.write(f"- **Decision threshold:** `{model.threshold}`")
    st.write(f"- **#Numeric features (XGB):** `{len(model.num_cols)}`")
    st.code(f"MODEL_PATH = {MODEL_PATH}")

tab1, tab2 = st.tabs(["Dự đoán 1 URL", "Dự đoán hàng loạt (CSV)"])

# ---- single URL ----
with tab1:
    url = st.text_input("Nhập URL để kiểm tra:", "", placeholder="vd: http://example.com/login")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Extract từ extract.py"):
            if not url.strip():
                st.warning("Hãy nhập URL.")
            else:
                feats = extract_record(url.strip(), label=None) or {}
                st.subheader("Features (từ data/extract.py)")
                st.dataframe(pd.DataFrame([feats]))

    with c2:
        if st.button("Predict"):
            if not url.strip():
                st.warning("Hãy nhập URL.")
            else:
                df_in = build_row_from_url(url.strip())
                proba = float(model.predict_proba_from_df(df_in)[0])
                pred = int(model.predict_from_df(df_in)[0])

                st.subheader("Prediction")
                st.metric("Probability (Phishing)", f"{proba:.4f}")
                st.write(f"**Prediction:** {'Phishing' if pred == 1 else 'Legitimate'} "
                         f"(threshold = {model.threshold})")

                with st.expander("Model input actually used"):
                    st.dataframe(df_in)

# ---- batch CSV ----
with tab2:
    st.write("Upload **CSV** có cột `URL` (hoặc `url`/`Url`). "
             "Nếu thiếu cột numeric, app sẽ tự trích xuất bằng `data/extract.py`.")
    up = st.file_uploader("Chọn file CSV", type=["csv"], accept_multiple_files=False)

    if up is not None:
        try:
            df_csv = pd.read_csv(up)
            st.subheader("Xem nhanh dữ liệu đầu vào")
            st.dataframe(df_csv.head(10))

            df_in = build_input_from_csv(df_csv)
            st.success(f"Đã chuẩn hoá dữ liệu: {df_in.shape[0]} dòng, {df_in.shape[1]} cột.")
            with st.expander("Model input actually used (preview)"):
                st.dataframe(df_in.head(10))

            # predict
            proba = model.predict_proba_from_df(df_in)
            preds = model.predict_from_df(df_in)

            out = df_csv.copy()
            out["p_blend"] = proba
            out["pred_blend"] = preds

            # download predictions
            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                " Tải predictions.csv",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )

            st.subheader("Thống kê")
            st.write(out[["p_blend", "pred_blend"]].describe())

        except Exception as e:
            st.error(f"Lỗi xử lý CSV: {type(e).__name__}: {e}")

st.caption("App dùng đúng extractor trong **data/extract.py**. "
           "Nếu CSV thiếu cột numeric so với `model.num_cols`, app sẽ tự bù từ extractor.")
