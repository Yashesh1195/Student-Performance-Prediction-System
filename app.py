import streamlit as st

from pages import (
    home,
    dataset_overview,
    data_analysis,
    prediction,
    model_info,
    batch_prediction,
)
from utils.data_loader import (
    load_data,
    load_splits,
    load_data_sample,
    load_splits_sample,
)
from utils.model_loader import load_model, load_scaler
from utils.preprocessing import build_label_encoders


def apply_theme() -> None:
    colors = {
        "bg": "#0B1020",
        "bg_grad": "linear-gradient(145deg, #0B1020 0%, #0F172A 45%, #0A0F1C 100%)",
        "surface": "#111827",
        "surface_alt": "#172338",
        "text": "#F8FAFC",
        "muted": "#B6C0D1",
        "accent": "#F6C945",
        "accent_soft": "rgba(246, 201, 69, 0.22)",
        "border": "rgba(255, 255, 255, 0.08)",
    }

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {{
        --bg: {colors["bg"]};
        --bg-grad: {colors["bg_grad"]};
        --surface: {colors["surface"]};
        --surface-alt: {colors["surface_alt"]};
        --text: {colors["text"]};
        --muted: {colors["muted"]};
        --accent: {colors["accent"]};
        --accent-soft: {colors["accent_soft"]};
        --border: {colors["border"]};
    }}

    .stApp {{
        background:
            radial-gradient(900px circle at 10% -10%, rgba(246, 201, 69, 0.08), transparent 55%),
            radial-gradient(800px circle at 80% 10%, rgba(96, 165, 250, 0.08), transparent 60%),
            repeating-linear-gradient(
                0deg,
                rgba(255, 255, 255, 0.03),
                rgba(255, 255, 255, 0.03) 1px,
                transparent 1px,
                transparent 26px
            ),
            var(--bg-grad);
        color: var(--text);
        font-family: 'IBM Plex Sans', sans-serif;
    }}

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'IBM Plex Sans', sans-serif;
        letter-spacing: 0.1px;
    }}

    .main .block-container {{
        padding-top: 2.5rem;
        animation: pageFade 0.6s ease-in-out;
    }}

    @keyframes pageFade {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    section[data-testid="stSidebar"] > div {{
        background: var(--surface);
        border-right: 1px solid var(--border);
    }}

    section[data-testid="stSidebar"] nav {{
        display: none;
    }}

    section[data-testid="stSidebarNav"] {{
        display: none;
    }}

    div[data-testid="stSidebarNav"] {{
        display: none;
    }}

    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    }}

    section[data-testid="stSidebar"] h2 {{
        font-size: 1.05rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 1rem;
    }}

    section[data-testid="stSidebar"] .stButton>button {{
        width: 100%;
        justify-content: flex-start;
        background: var(--surface_alt);
        color: var(--text);
        border: 1px solid transparent;
        border-radius: 14px;
        padding: 0.65rem 0.9rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }}

    section[data-testid="stSidebar"] .stButton>button:hover {{
        border-color: var(--accent);
        background: rgba(242, 184, 75, 0.12);
        transform: translateX(4px);
    }}

    section[data-testid="stSidebar"] .stButton>button[kind="primary"] {{
        border-color: var(--accent);
        background: linear-gradient(120deg, rgba(242, 184, 75, 0.25), rgba(20, 30, 43, 0.9));
        box-shadow: 0 0 0 1px var(--accent_soft);
    }}


    div[data-testid="stMetric"] label {{
        color: var(--muted);
    }}

    .stTabs [data-baseweb="tab"] {{
        font-weight: 600;
    }}

    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: var(--accent);
        border-bottom: 2px solid var(--accent);
    }}

    .stButton>button, .stDownloadButton>button {{
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.1rem;
        font-weight: 600;
    }}

    .stButton>button:hover, .stDownloadButton>button:hover {{
        filter: brightness(0.95);
    }}

    .accent-card {{
        background: var(--surface-alt);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


PAGES = {
    "Home": {
        "icon": "🎓",
        "module": home,
    },
    "Dataset Overview": {
        "icon": "📘",
        "module": dataset_overview,
    },
    "Data Analysis": {
        "icon": "📊",
        "module": data_analysis,
    },
    "Prediction": {
        "icon": "🧮",
        "module": prediction,
    },
    "Model Info": {
        "icon": "📄",
        "module": model_info,
    },
    "Batch Prediction": {
        "icon": "📑",
        "module": batch_prediction,
    },
}


st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

apply_theme()

with st.sidebar:
    st.markdown("## Navigation")
    if "nav" not in st.session_state:
        st.session_state.nav = next(iter(PAGES.keys()))

    for page_name, page_meta in PAGES.items():
        is_active = st.session_state.nav == page_name
        if st.button(
            f"{page_meta['icon']}  {page_name}",
            key=f"nav-{page_name}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            if st.session_state.nav != page_name:
                st.session_state.nav = page_name
                st.rerun()

page_key = st.session_state.nav

if "data_sample" not in st.session_state:
    try:
        st.session_state.data_sample = load_data_sample()
        st.session_state.data = st.session_state.data_sample
    except Exception as exc:
        st.error(f"Data loading failed: {exc}")

if "splits_sample" not in st.session_state:
    try:
        st.session_state.splits_sample = load_splits_sample()
        st.session_state.splits = st.session_state.splits_sample
    except Exception as exc:
        st.session_state.splits_sample = {}
        st.session_state.splits = {}
        st.warning(f"Split data unavailable: {exc}")

if "reference_df" not in st.session_state:
    st.session_state.reference_df = st.session_state.splits_sample.get(
        "train", st.session_state.get("data_sample")
    )

if "encoders" not in st.session_state and st.session_state.get("reference_df") is not None:
    try:
        st.session_state.encoders = build_label_encoders(st.session_state.reference_df)
    except Exception as exc:
        st.error(f"Encoder setup failed: {exc}")
        st.stop()

if "model" not in st.session_state:
    try:
        st.session_state.model = load_model()
    except Exception as exc:
        st.error(f"Model loading failed: {exc}")

if "scaler" not in st.session_state:
    st.session_state.scaler = load_scaler()

PAGES[page_key]["module"].render()
