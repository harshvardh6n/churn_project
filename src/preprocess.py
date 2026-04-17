"""
preprocess.py
=============
Cleaning and preprocessing pipeline for the Telco Churn dataset.

Steps
-----
1. Drop customerID (non-predictive identifier)
2. Fix TotalCharges (stored as object due to spaces → cast to float)
3. Impute missing TotalCharges with median
4. Encode binary target (Churn: Yes→1, No→0)
5. Return X (features) and y (target)
"""

import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ── Column lists ───────────────────────────────────────────────────────────────
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS  = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]
CATEGORICAL_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]
DROP_COLS = ["customerID"]
TARGET_COL = "Churn"


# ── Cleaning helpers ───────────────────────────────────────────────────────────

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Cast TotalCharges to float; coerce errors → NaN (11 affected rows)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"].fillna(median_val, inplace=True)
        logger.info(
            "TotalCharges: imputed %d missing values with median (%.2f)",
            n_missing, median_val,
        )
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No and gender to 0/1 integers."""
    df = df.copy()
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline applied to raw DataFrame.

    Parameters
    ----------
    df : pd.DataFrame  — raw dataset from data_loader.load_raw_data()

    Returns
    -------
    pd.DataFrame  — cleaned dataset
    """
    logger.info("Starting cleaning pipeline on %d rows …", len(df))
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = fix_total_charges(df)
    df = encode_binary_columns(df)
    df = df.drop_duplicates()
    logger.info("Cleaned dataset: %d rows × %d columns", *df.shape)
    return df


def split_xy(df: pd.DataFrame):
    """Return X (features) and y (0/1 target)."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


# ── sklearn ColumnTransformer ──────────────────────────────────────────────────

def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer.

    Numeric  → median impute → standard scale
    Categorical → constant impute → one-hot encode (drop='first')
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="passthrough",   # keep binary columns as-is
    )
    return preprocessor


# ── Feature name helper ────────────────────────────────────────────────────────

def get_feature_names(preprocessor: ColumnTransformer, num_cols, cat_cols, passthrough_cols) -> list:
    """Extract final feature names after ColumnTransformer is fitted."""
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["encoder"]
        .get_feature_names_out(cat_cols)
    )
    return list(num_cols) + cat_names + list(passthrough_cols)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_raw_data

    raw = load_raw_data()
    df  = clean(raw)
    X, y = split_xy(df)

    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts())
    print("\nNumeric cols:", NUMERIC_COLS)
    print("Categorical cols:", CATEGORICAL_COLS)