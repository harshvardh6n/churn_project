"""
feature_engineering.py
=======================
Domain-driven feature creation for the Telco Churn dataset.

New Features
------------
AvgCharges        - MonthlyCharges / (tenure + 1)          [charge efficiency]
TenureGroup       - Ordinal bucket: new / growing / loyal / champion
ServiceCount      - # of value-added services subscribed
HasHighValuePlan  - 1 if Contract is One/Two year
ChargeRatio       - MonthlyCharges / (TotalCharges + 1)    [early vs mature]
IsSeniorAlone     - Senior with no partner & no dependents
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Feature creation functions ─────────────────────────────────────────────────

def add_avg_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Average monthly spend per tenure month."""
    df = df.copy()
    df["AvgCharges"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    logger.debug("Added AvgCharges")
    return df


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket tenure into 4 life-cycle stages (encoded as 0–3)."""
    df = df.copy()
    bins   = [0, 12, 24, 48, 10_000]
    labels = [0, 1, 2, 3]          # new / growing / loyal / champion
    # FIXED
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, right=True
    ).astype("object").fillna(0).astype(int)
    logger.debug("Added TenureGroup")
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count of add-on services the customer has subscribed to."""
    df = df.copy()
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    # Map Yes→1, everything else→0
    def _yes(series):
        if series.dtype == object:
            return (series == "Yes").astype(int)
        return series.clip(0, 1)

    df["ServiceCount"] = sum(_yes(df[c]) for c in service_cols if c in df.columns)
    logger.debug("Added ServiceCount")
    return df


def add_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio of current monthly charge to lifetime total - high ratio = recent joiner."""
    df = df.copy()
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    logger.debug("Added ChargeRatio")
    return df


def add_high_value_plan(df: pd.DataFrame) -> pd.DataFrame:
    """Flag customers on long-term contracts (less likely to churn)."""
    df = df.copy()
    if df["Contract"].dtype == object:
        df["HasHighValuePlan"] = df["Contract"].isin(["One year", "Two year"]).astype(int)
    else:
        # already encoded as 0/1/2 in some pipelines
        df["HasHighValuePlan"] = (df["Contract"] > 0).astype(int)
    logger.debug("Added HasHighValuePlan")
    return df


def add_senior_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Senior citizen with no partner and no dependents - higher risk group."""
    df = df.copy()
    # Works whether columns are 0/1 int or Yes/No strings
    def _is_yes(col):
        if df[col].dtype == object:
            return (df[col] == "Yes").astype(int)
        return df[col].fillna(0).astype(int)

    senior = df["SeniorCitizen"].fillna(0).astype(int)
    partner    = _is_yes("Partner")
    dependents = _is_yes("Dependents")
    df["IsSeniorAlone"] = ((senior == 1) & (partner == 0) & (dependents == 0)).astype(int)
    logger.debug("Added IsSeniorAlone")
    return df


# ── Master pipeline ────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in order.

    Parameters
    ----------
    df : pd.DataFrame - cleaned dataset (output of preprocess.clean)

    Returns
    -------
    pd.DataFrame - enriched dataset
    """
    logger.info("Engineering features on %d rows …", len(df))
    df = add_avg_charges(df)
    df = add_tenure_group(df)
    df = add_service_count(df)
    df = add_charge_ratio(df)
    df = add_high_value_plan(df)
    df = add_senior_alone(df)

    new_cols = [
        "AvgCharges", "TenureGroup", "ServiceCount",
        "ChargeRatio", "HasHighValuePlan", "IsSeniorAlone",
    ]
    logger.info(
        "Feature engineering complete - %d new columns added: %s",
        len(new_cols), new_cols,
    )
    return df


ENGINEERED_NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "AvgCharges", "ChargeRatio", "ServiceCount",
]
ENGINEERED_PASSTHROUGH_COLS = [
    "SeniorCitizen", "gender", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling",
    "TenureGroup", "HasHighValuePlan", "IsSeniorAlone",
]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_raw_data
    from preprocess   import clean

    raw = load_raw_data()
    df  = engineer_features(clean(raw))
    print(df[["AvgCharges","TenureGroup","ServiceCount","ChargeRatio",
              "HasHighValuePlan","IsSeniorAlone"]].head(10))
    print("\nShape:", df.shape)