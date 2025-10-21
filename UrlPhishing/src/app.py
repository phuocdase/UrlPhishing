# app.py
import streamlit as st
import joblib, json
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
  # bạn lưu hàm extract_features ra file feature_extractor.py

# ===== Load artifacts =====
MODEL_PATH = r"E:\KLTN\UrlPhishing\src\artifacts_lr_tfidf\logreg_calibrated.joblib"
SCALER_PATH = r"E:\KLTN\UrlPhishing\src\artifacts_lr_tfidf\tfidf_vectorizer.joblib"
FEATURE_PATH = "artifacts_xgb/feature_order.json"
METRICS_PATH = "artifacts_xgb/metrics.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURE_PATH) as f: feature_order = json.load(f)
with open(METRICS_PATH) as f: metrics = json.load(f)

# ép model về CPU
if isinstance(model, CalibratedClassifierCV):
    for cc in model.calibrated_classifiers_:
        if isinstance(cc.estimator, xgb.XGBClassifier):
            cc.estimator.set_params(device="cpu")
import re
import tldextract
from urllib.parse import urlparse


def extract_features(url: str) -> dict:
    features = {}

    # chuẩn hóa URL
    u = url if url.startswith(("http://", "https://")) else "http://" + url
    parsed = urlparse(u)
    domain = parsed.netloc
    path = parsed.path or ""
    ext = tldextract.extract(u)

    # ===== 1. Thông tin cơ bản =====
    features["URL"] = url
    features["LengthOfURL"] = len(u)
    features["Domain"] = domain
    features["DomainLengthOfURL"] = len(domain)

    # Domain là IP?
    features["IsDomainIP"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$", domain) else 0

    # TLD length
    features["TLDLength"] = len(ext.suffix)

    # Số subdomain
    features["NumberOfSubdomains"] = len(ext.subdomain.split(".")) if ext.subdomain else 0

    # ===== 2. Đếm ký tự =====
    letters = sum(c.isalpha() for c in u)
    digits = sum(c.isdigit() for c in u)
    specials = sum(not c.isalnum() for c in u)

    features["LetterCntInURL"] = letters
    features["DigitCntInURL"] = digits
    features["URLLetterRatio"] = letters / features["LengthOfURL"] if features["LengthOfURL"] > 0 else 0
    features["URLDigitRatio"] = digits / features["LengthOfURL"] if features["LengthOfURL"] > 0 else 0
    features["OtherSpclCharCntInURL"] = specials
    features["URLOtherSpclCharRatio"] = specials / features["LengthOfURL"] if features["LengthOfURL"] > 0 else 0

    # ===== 3. Ký tự đặc biệt cụ thể =====
    features["EqualCharCntInURL"] = u.count("=")
    features["QuesMarkCntInURL"] = u.count("?")
    features["AmpCharCntInURL"] = u.count("&")

    # ===== 4. Pattern đặc biệt =====
    features["HexPatternCnt"] = len(re.findall(r"0x[0-9a-fA-F]+", u))
    features["Base64PatternCnt"] = 1 if re.search(r"(?:[A-Za-z0-9+/]{8,}={0,2})", u) else 0

    # ===== 5. URL Complexity (theo dataset: chia cho 8, bội số 0.125) =====
    features["URLComplexity"] = (features["DomainLengthOfURL"] +
                                 features["OtherSpclCharCntInURL"] +
                                 features["NumberOfSubdomains"]) / 8

    return features


# ===== UI =====
st.title("🔎 URL Phishing Detection (XGBoost with Dataset-style Features)")
st.json(metrics)

url_input = st.text_input("Nhập URL để kiểm tra:", "")

if st.button("Extract & Predict") and url_input:
    # 1. Extract features
    feats = extract_features(url_input)
    df_feats = pd.DataFrame([feats])

    # 2. Show extracted features
    st.subheader("📊 Extracted Features")
    st.dataframe(df_feats)

    # 3. Reorder features & scale
    X = df_feats[feature_order]
    X_scaled = scaler.transform(X)

    # 4. Predict
    proba = model.predict_proba(X_scaled)[0,1]
    pred = int(proba >= metrics.get("best_threshold", 0.5))

    st.subheader("🔮 Prediction")
    st.write(f"**Prediction:** {'🚨 Phishing' if pred==1 else '✅ Legitimate'}")
    st.write(f"**Probability (Phishing):** {proba:.4f}")
