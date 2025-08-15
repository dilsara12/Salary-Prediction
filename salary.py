import streamlit as st
import pandas as pd
import numpy as np
import pickle
from typing import List

st.set_page_config(page_title="AI Job Salary Predictor", page_icon="💼", layout="centered")

# ------------------------------
# Load artifacts (from pickle)
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(path: str = "models_best_salary.pkl"):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # If a raw estimator was saved by mistake, wrap it so app still runs
    if hasattr(obj, "predict"):
        return {
            "best_model": obj,
            "best_model_name": obj.__class__.__name__,
            "ohe": None,
            "mlb": None,
            "feature_order": None,
            "scaler_for_ridge": None,
            "cat_levels": {},
        }

    if not isinstance(obj, dict):
        raise TypeError("Unsupported pickle format. Expected dict or estimator.")

    # Handle nested dict models like {'best_model': {'model': estimator}}
    bm = obj.get("best_model")
    if isinstance(bm, dict):
        for k in ("model", "estimator", "pipe", "clf"):
            if k in bm and hasattr(bm[k], "predict"):
                obj["best_model"] = bm[k]
                break

    # Basic checks
    if not hasattr(obj.get("best_model", None), "predict"):
        raise ValueError("No estimator with .predict found in artifacts['best_model'].")

    needed = ["ohe", "mlb", "feature_order", "scaler_for_ridge", "cat_levels"]
    for k in needed:
        if k not in obj:
            raise KeyError(f"Artifacts missing '{k}'. Re-export from your training notebook.")
    return obj

A = load_artifacts("models_best_salary.pkl")

# ------------------------------
# Helpers: region + feature build
# ------------------------------
def to_region(cc: str) -> str:
    cc = str(cc).upper()
    AMER = {'US','CA','MX','BR','AR','CL'}
    EMEA = {'UK','DE','FR','ES','IT','NL','SE','NO','PL','AE','SA','EG','NG','KE','ZA'}
    APAC = {'IN','CN','JP','KR','SG','AU','NZ','LK','BD','TH','VN','PH','MY'}
    if cc in AMER: return 'AMER'
    if cc in EMEA: return 'EMEA'
    if cc in APAC: return 'APAC'
    return 'OTHER'

def build_features_single(
    experience_level: str,
    employment_type: str,
    remote_status: str,
    country_code: str,
    role: str,
    skills_list: List[str],
    posted_year: int
) -> pd.DataFrame:
    """Recreate training-time feature matrix for a single example."""
    if A["ohe"] is None or A["mlb"] is None or A["feature_order"] is None:
        raise RuntimeError("Artifacts missing encoders/feature_order. Re-export from training script.")

    region = to_region(country_code)

    # Categorical
    cat_df = pd.DataFrame([{
        "Experience_Level": experience_level,
        "Employment_Type":  employment_type,
        "Remote_Status":    remote_status,
        "Country":          country_code,
        "Role":             role,
        "Region":           region
    }])
    ohe = A["ohe"]
    ohe_cols = ohe.get_feature_names_out(["Experience_Level","Employment_Type","Remote_Status","Country","Role","Region"])
    ohe_df = pd.DataFrame(ohe.transform(cat_df), columns=ohe_cols, index=[0])

    # Skills (only those seen during training are used)
    mlb = A["mlb"]
    skills_keep = [s for s in skills_list if s in mlb.classes_]
    skills_df = pd.DataFrame(
        mlb.transform([skills_keep]),
        columns=[f"skill_{s}" for s in mlb.classes_],
        index=[0]
    )

    # Numeric
    num_df = pd.DataFrame({"Posted_Year": [int(posted_year)]})

    # Combine & reorder columns to match training
    X = pd.concat([skills_df, ohe_df, num_df], axis=1)
    X = X.reindex(columns=A["feature_order"], fill_value=0)
    return X

def predict_salary(X_one: pd.DataFrame) -> float:
    model = A["best_model"]
    # Apply scaler if your best model used it (Ridge in our training)
    if A.get("best_model_name") == "Ridge" and A.get("scaler_for_ridge") is not None:
        X_one = A["scaler_for_ridge"].transform(X_one)
    y = model.predict(X_one)
    return float(np.ravel(y)[0])

# ------------------------------
# UI
# ------------------------------
st.title("AI Job Salary Predictor")
st.caption("Predict salaries for AI & Data roles using your trained model.")

cat_levels = A.get("cat_levels", {})
skills_all = list(A["mlb"].classes_) if A.get("mlb") is not None else []

col1, col2 = st.columns(2)
with col1:
    experience_level = st.selectbox(
        "Experience Level",
        options=cat_levels.get("Experience_Level", ["Entry","Junior","Mid","Senior","Expert"]),
        index=0
    )
    employment_type = st.selectbox(
        "Employment Type",
        options=cat_levels.get("Employment_Type", ["Full-time","Part-time","Contract","Internship"])
    )
    posted_year = st.number_input("Posting Year", min_value=2018, max_value=2035, value=2025, step=1)

with col2:
    remote_status = st.selectbox(
        "Work Arrangement",
        options=cat_levels.get("Remote_Status", ["Onsite","Hybrid","Remote"])
    )
    country_code = st.selectbox(
        "Country",
        options=cat_levels.get("Country", ["US","UK","IN","CA","JP","DE","FR","SG","AU"])
    )
    role = st.selectbox(
        "Job Role",
        options=cat_levels.get("Role", ["Data Scientist","ML Engineer","AI Engineer","Data Analyst","Research Scientist"])
    )

skills_default = ["Python","SQL"] if {"Python","SQL"}.issubset(skills_all) else skills_all[:2]
skills = st.multiselect("Key Skills", options=sorted(skills_all), default=skills_default)

st.write("---")
if st.button("Predict Salary"):
    try:
        X_one = build_features_single(
            experience_level=experience_level,
            employment_type=employment_type,
            remote_status=remote_status,
            country_code=country_code,
            role=role,
            skills_list=skills,
            posted_year=int(posted_year),
        )
        y_hat = predict_salary(X_one)
        st.success(f"Estimated Salary (USD): **${y_hat:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.expander("Debug info (artifacts)"):
    st.write({
        "best_model_name": A.get("best_model_name"),
        "best_model_type": type(A.get("best_model")).__name__,
        "#features": len(A.get("feature_order", [])),
        "#skills_known": len(getattr(A["mlb"], "classes_", [])),
    })
