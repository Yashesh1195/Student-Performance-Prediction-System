from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

MODEL_CANDIDATES = [
    ROOT / "model" / "model.pkl",
    ROOT / "student_performance_best_model.pkl",
]

SCALER_CANDIDATES = [
    ROOT / "model" / "scaler.pkl",
    ROOT / "student_performance_scaler.pkl",
]


@st.cache_resource(show_spinner=False)
def load_model():
    for path in MODEL_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(
        "Model file not found. Expected model/model.pkl or student_performance_best_model.pkl."
    )


@st.cache_resource(show_spinner=False)
def load_scaler():
    for path in SCALER_CANDIDATES:
        if path.exists():
            return joblib.load(path)
    return None
