from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder

TARGET_COL = "GPA"

MAP_COLS = {
    "Gender": {"Female": 0, "Male": 1},
    "SchoolType": {"Public": 0, "Private": 1},
}

LABEL_ENCODE_COLS = ["Race", "ParentalEducation", "Locale"]


def _summarize_columns(columns: List[str], limit: int = 20) -> str:
    if len(columns) <= limit:
        return ", ".join(columns)
    return ", ".join(columns[:limit]) + f", ... (+{len(columns) - limit} more)"


def get_feature_columns(reference_df: pd.DataFrame) -> List[str]:
    return [col for col in reference_df.columns if col != TARGET_COL]


def build_label_encoders(reference_df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    encoders: Dict[str, LabelEncoder] = {}
    missing = [col for col in LABEL_ENCODE_COLS if col not in reference_df.columns]
    if missing:
        available = _summarize_columns([str(col) for col in reference_df.columns])
        missing_list = ", ".join(missing)
        raise ValueError(
            "Missing expected columns for label encoding: "
            f"{missing_list}. Available columns: {available}. "
            "Verify the CSV schema and remote file access."
        )
    for col in LABEL_ENCODE_COLS:
        le = LabelEncoder()
        values = reference_df[col].astype(str).dropna().unique()
        le.fit(values)
        encoders[col] = le
    return encoders


def _apply_mappings(df: pd.DataFrame) -> pd.DataFrame:
    for col, mapping in MAP_COLS.items():
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        df[col] = df[col].map(mapping)
    return df


def _apply_label_encoders(
    df: pd.DataFrame, encoders: Dict[str, LabelEncoder]
) -> pd.DataFrame:
    for col, encoder in encoders.items():
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        values = df[col].astype(str)
        unseen = set(values.unique()) - set(encoder.classes_)
        if unseen:
            raise ValueError(
                f"Unseen categories in {col}: {', '.join(sorted(unseen))}"
            )
        df[col] = encoder.transform(values)
    return df


def validate_columns(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[List[str], List[str]]:
    missing = [col for col in feature_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in feature_columns and col != TARGET_COL]
    return missing, extra


def preprocess_input(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    scaler=None,
    encoders: Dict[str, LabelEncoder] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if reference_df is None:
        raise ValueError("Reference dataset is required for preprocessing.")

    working = df.copy()
    if TARGET_COL in working.columns:
        working = working.drop(columns=[TARGET_COL])

    feature_columns = get_feature_columns(reference_df)
    missing, _ = validate_columns(working, feature_columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    working = working[feature_columns]
    working = _apply_mappings(working)
    if encoders is None:
        encoders = build_label_encoders(reference_df)
    working = _apply_label_encoders(working, encoders)

    for col in working.columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    if working.isnull().any().any():
        raise ValueError("Input contains invalid or missing values after preprocessing.")

    if scaler is not None:
        transformed = scaler.transform(working)
        return pd.DataFrame(transformed, columns=feature_columns), feature_columns

    return working, feature_columns


def split_features_target(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    scaler=None,
    encoders: Dict[str, LabelEncoder] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if TARGET_COL not in df.columns:
        raise ValueError("Target column GPA not found in dataset.")

    features, feature_columns = preprocess_input(
        df.drop(columns=[TARGET_COL]),
        reference_df,
        scaler=scaler,
        encoders=encoders,
    )
    target = df[TARGET_COL]
    return features, target, feature_columns
