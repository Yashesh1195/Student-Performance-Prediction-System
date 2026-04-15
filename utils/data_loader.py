from pathlib import Path
import io
import os
from urllib.parse import parse_qs, urlparse

import pandas as pd
import streamlit as st

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASETS_DIR = ROOT / "datasets"
DEFAULT_SAMPLE_ROWS = 20000


def _get_config_value(key: str) -> str | None:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return os.getenv(key)


def _extract_drive_file_id(value: str | None) -> str | None:
    if not value:
        return None

    if "drive.google.com" not in value and "docs.google.com" not in value:
        if "/" not in value and " " not in value:
            return value
        return None

    parsed = urlparse(value)
    query_id = parse_qs(parsed.query).get("id", [None])[0]
    if query_id:
        return query_id

    if "/d/" in parsed.path:
        return parsed.path.split("/d/")[1].split("/")[0]

    return None


def _get_drive_confirm_token(response: "requests.Response") -> str | None: # type: ignore
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _download_drive_file(file_id: str) -> io.BytesIO:
    if requests is None:
        raise RuntimeError("requests is required to download Google Drive files.")

    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    response = session.get(base_url, params={"id": file_id}, stream=True)
    token = _get_drive_confirm_token(response)
    if token:
        response = session.get(
            base_url, params={"id": file_id, "confirm": token}, stream=True
        )

    response.raise_for_status()
    buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            buffer.write(chunk)
    buffer.seek(0)
    return buffer


def _read_csv_from_source(value: str, *, nrows: int | None = None) -> pd.DataFrame:
    file_id = _extract_drive_file_id(value)
    if file_id:
        buffer = _download_drive_file(file_id)
        return pd.read_csv(buffer, nrows=nrows)

    return pd.read_csv(value, nrows=nrows)


def _load_remote_split(split_name: str, *, nrows: int | None = None) -> pd.DataFrame | None:
    key = f"{split_name.upper()}_URL"
    value = _get_config_value(key)
    if not value:
        value = _get_config_value(f"GDRIVE_{split_name.upper()}_ID")

    if not value:
        return None

    df_split = _read_csv_from_source(value, nrows=nrows)
    df_split["Split"] = split_name
    return df_split


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    primary_path = DATA_DIR / "StudentsPerformance.csv"
    if primary_path.exists():
        return pd.read_csv(primary_path)

    splits = []
    for split_name in ["train", "test", "validation"]:
        split_path = DATASETS_DIR / f"{split_name}.csv"
        if split_path.exists():
            df_split = pd.read_csv(split_path)
            df_split["Split"] = split_name
            splits.append(df_split)

    if not splits:
        for split_name in ["train", "test", "validation"]:
            df_split = _load_remote_split(split_name)
            if df_split is not None:
                splits.append(df_split)

    if not splits:
        remote_dataset = _get_config_value("DATASET_URL") or _get_config_value(
            "GDRIVE_DATASET_ID"
        )
        if remote_dataset:
            return _read_csv_from_source(remote_dataset)

        raise FileNotFoundError(
            "No dataset found locally or in remote configuration (DATASET_URL/TRAIN_URL)."
        )

    return pd.concat(splits, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_data_sample(max_rows: int = DEFAULT_SAMPLE_ROWS) -> pd.DataFrame:
    primary_path = DATA_DIR / "StudentsPerformance.csv"
    if primary_path.exists():
        return pd.read_csv(primary_path, nrows=max_rows)

    splits = []
    for split_name in ["train", "test", "validation"]:
        split_path = DATASETS_DIR / f"{split_name}.csv"
        if split_path.exists():
            df_split = pd.read_csv(split_path, nrows=max_rows)
            df_split["Split"] = split_name
            splits.append(df_split)

    if not splits:
        for split_name in ["train", "test", "validation"]:
            df_split = _load_remote_split(split_name, nrows=max_rows)
            if df_split is not None:
                splits.append(df_split)

    if not splits:
        remote_dataset = _get_config_value("DATASET_URL") or _get_config_value(
            "GDRIVE_DATASET_ID"
        )
        if remote_dataset:
            return _read_csv_from_source(remote_dataset, nrows=max_rows)

        raise FileNotFoundError(
            "No dataset found locally or in remote configuration (DATASET_URL/TRAIN_URL)."
        )

    return pd.concat(splits, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_splits() -> dict:
    splits = {}
    for split_name in ["train", "test", "validation"]:
        split_path = DATASETS_DIR / f"{split_name}.csv"
        if split_path.exists():
            splits[split_name] = pd.read_csv(split_path)

    if splits:
        return splits

    for split_name in ["train", "test", "validation"]:
        df_split = _load_remote_split(split_name)
        if df_split is not None:
            splits[split_name] = df_split

    if splits:
        return splits

    primary_path = DATA_DIR / "StudentsPerformance.csv"
    if primary_path.exists():
        splits["all"] = pd.read_csv(primary_path)

    if splits:
        return splits

    remote_dataset = _get_config_value("DATASET_URL") or _get_config_value(
        "GDRIVE_DATASET_ID"
    )
    if remote_dataset:
        splits["all"] = _read_csv_from_source(remote_dataset)

    return splits


@st.cache_data(show_spinner=False)
def load_splits_sample(max_rows: int = DEFAULT_SAMPLE_ROWS) -> dict:
    splits = {}
    for split_name in ["train", "test", "validation"]:
        split_path = DATASETS_DIR / f"{split_name}.csv"
        if split_path.exists():
            splits[split_name] = pd.read_csv(split_path, nrows=max_rows)

    if splits:
        return splits

    for split_name in ["train", "test", "validation"]:
        df_split = _load_remote_split(split_name, nrows=max_rows)
        if df_split is not None:
            splits[split_name] = df_split

    if splits:
        return splits

    primary_path = DATA_DIR / "StudentsPerformance.csv"
    if primary_path.exists():
        splits["all"] = pd.read_csv(primary_path, nrows=max_rows)

    if splits:
        return splits

    remote_dataset = _get_config_value("DATASET_URL") or _get_config_value(
        "GDRIVE_DATASET_ID"
    )
    if remote_dataset:
        splits["all"] = _read_csv_from_source(remote_dataset, nrows=max_rows)

    return splits
