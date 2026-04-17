"""
data_loader.py
==============
Handles dataset download, caching, and loading for the
Customer Churn Prediction project.

Dataset: IBM Telco Customer Churn (7043 rows × 21 columns)
Source : Public CSV mirror (no Kaggle auth required)
"""

import os
import logging
import requests
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Public mirror of the Telco dataset
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(RAW_DIR, "telco_churn_raw.csv")


# ── Public API ─────────────────────────────────────────────────────────────────

def download_dataset(force: bool = False) -> str:
    """Download raw CSV to *data/* if it doesn't already exist.

    Parameters
    ----------
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    str
        Absolute path to the downloaded CSV.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_PATH) and not force:
        logger.info("Dataset already present at %s - skipping download.", RAW_PATH)
        return RAW_PATH

    logger.info("Downloading dataset from %s …", DATASET_URL)
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()

    with open(RAW_PATH, "wb") as fh:
        fh.write(response.content)

    logger.info("Saved %d bytes → %s", len(response.content), RAW_PATH)
    return RAW_PATH


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the raw Telco CSV into a DataFrame.

    Parameters
    ----------
    path : str, optional
        Override the default path.

    Returns
    -------
    pd.DataFrame
        Raw dataset with no transformations applied.
    """
    target_path = path or RAW_PATH

    if not os.path.exists(target_path):
        logger.warning("Raw file not found - triggering download.")
        download_dataset()

    df = pd.read_csv(target_path)
    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)
    return df


def get_feature_metadata() -> dict:
    """Return a human-readable description of every column."""
    return {
        "customerID":        "Unique customer identifier",
        "gender":            "Male / Female",
        "SeniorCitizen":     "1 = Senior (65+), 0 = Not senior",
        "Partner":           "Has a partner: Yes / No",
        "Dependents":        "Has dependents: Yes / No",
        "tenure":            "Months the customer has been with the company",
        "PhoneService":      "Has phone service: Yes / No",
        "MultipleLines":     "Has multiple phone lines",
        "InternetService":   "DSL / Fiber optic / No",
        "OnlineSecurity":    "Has online security add-on",
        "OnlineBackup":      "Has online backup add-on",
        "DeviceProtection":  "Has device protection add-on",
        "TechSupport":       "Has tech support add-on",
        "StreamingTV":       "Has streaming TV add-on",
        "StreamingMovies":   "Has streaming movies add-on",
        "Contract":          "Month-to-month / One year / Two year",
        "PaperlessBilling":  "Uses paperless billing: Yes / No",
        "PaymentMethod":     "Electronic check / Mailed check / …",
        "MonthlyCharges":    "Current monthly charges ($)",
        "TotalCharges":      "Total charges billed to date ($)",
        "Churn":             "TARGET - Did the customer churn? Yes / No",
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = download_dataset()
    df   = load_raw_data(path)
    print(df.head())
    print("\nShape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])