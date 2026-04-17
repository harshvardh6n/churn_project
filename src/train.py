"""
train.py
========
End-to-end model training with:
  • sklearn Pipeline (preprocessor → model)
  • SMOTE for class imbalance
  • Three candidate models (LR, RF, XGBoost)
  • RandomizedSearchCV hyperparameter tuning
  • 5-fold stratified cross-validation
  • Model serialisation to models/

Run:
    python src/train.py
"""

import os, sys, logging, json, warnings
import numpy  as np
import pandas as pd
import joblib

from sklearn.model_selection    import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import roc_auc_score, f1_score, accuracy_score
from imblearn.over_sampling     import SMOTE
from imblearn.pipeline          import Pipeline as ImbPipeline
from xgboost                    import XGBClassifier

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from data_loader         import load_raw_data, download_dataset
from preprocess          import clean, split_xy, build_preprocessor, CATEGORICAL_COLS
from feature_engineering import engineer_features, ENGINEERED_NUMERIC_COLS, ENGINEERED_PASSTHROUGH_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Hyperparameter grids ───────────────────────────────────────────────────────
LR_GRID = {
    "clf__C":            [0.01, 0.1, 1.0, 10.0],
    "clf__solver":       ["lbfgs", "saga"],
    "clf__max_iter":     [500, 1000],
    "clf__class_weight": ["balanced", None],
}

RF_GRID = {
    "clf__n_estimators":      [100, 200, 300],
    "clf__max_depth":         [4, 6, 8, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__class_weight":      ["balanced", None],
}

XGB_GRID = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__max_depth":     [3, 4, 5, 6],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__subsample":     [0.7, 0.8, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 1.0],
    "clf__scale_pos_weight": [1, 2, 3],
}


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(model, numeric_cols, categorical_cols) -> ImbPipeline:
    """Preprocessor → SMOTE → Model.
    SMOTE must come AFTER the preprocessor — it requires all-numeric input.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    return ImbPipeline([
        ("prep",  preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("clf",   model),
    ])
# ── Training function ─────────────────────────────────────────────────────────

def train_and_tune(pipeline, param_grid, X_train, y_train, n_iter=20):
    """RandomizedSearchCV with stratified 5-fold CV, scoring on AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator  = pipeline,
        param_distributions = param_grid,
        n_iter     = n_iter,
        scoring    = "roc_auc",
        cv         = cv,
        n_jobs     = -1,
        random_state = 42,
        verbose    = 1,
        refit      = True,
    )
    search.fit(X_train, y_train)
    return search


# ── Evaluation helper ─────────────────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results = {
        "model":    name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1":       round(f1_score(y_test, y_pred), 4),
        "auc":      round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info(
        "%-24s  AUC=%.4f  F1=%.4f  Acc=%.4f",
        name, results["auc"], results["f1"], results["accuracy"],
    )
    return results


# ── Main training routine ─────────────────────────────────────────────────────

def main():
    # 1. Load & prepare data
    logger.info("Loading dataset …")
    download_dataset()
    raw = load_raw_data()
    df  = engineer_features(clean(raw))
    X, y = split_xy(df)

    NUM_COLS  = ENGINEERED_NUMERIC_COLS
    CAT_COLS  = [c for c in CATEGORICAL_COLS if c in X.columns]
    PASS_COLS = [c for c in ENGINEERED_PASSTHROUGH_COLS if c in X.columns]

    logger.info("Target distribution:\n%s", y.value_counts().to_string())

    # 2. Train / test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # ── Save test set for evaluation page ────────────────────────────────────
    test_df = X_test.copy()
    test_df["Churn"] = y_test.values
    test_df.to_csv(os.path.join(ROOT_DIR, "data", "test_set.csv"), index=False)
    logger.info("Test set saved.")

    results_all = []

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    logger.info("\n=== Logistic Regression ===")
    lr_pipe   = build_pipeline(
        LogisticRegression(random_state=42), NUM_COLS, CAT_COLS
    )
    lr_search = train_and_tune(lr_pipe, LR_GRID, X_train, y_train, n_iter=12)
    lr_result = evaluate_model("Logistic Regression", lr_search.best_estimator_, X_test, y_test)
    results_all.append(lr_result)
    joblib.dump(lr_search.best_estimator_, os.path.join(MODELS_DIR, "logistic_regression.pkl"))

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    logger.info("\n=== Random Forest ===")
    rf_pipe   = build_pipeline(
        RandomForestClassifier(random_state=42, n_jobs=-1), NUM_COLS, CAT_COLS
    )
    rf_search = train_and_tune(rf_pipe, RF_GRID, X_train, y_train, n_iter=15)
    rf_result = evaluate_model("Random Forest", rf_search.best_estimator_, X_test, y_test)
    results_all.append(rf_result)
    joblib.dump(rf_search.best_estimator_, os.path.join(MODELS_DIR, "random_forest.pkl"))

    # ── Model 3: XGBoost (best) ───────────────────────────────────────────────
    logger.info("\n=== XGBoost ===")
    xgb_pipe = build_pipeline(
        XGBClassifier(
            random_state=42, eval_metric="logloss",
            use_label_encoder=False, tree_method="hist",
        ),
        NUM_COLS, CAT_COLS,
    )
    xgb_search = train_and_tune(xgb_pipe, XGB_GRID, X_train, y_train, n_iter=20)
    xgb_result = evaluate_model("XGBoost", xgb_search.best_estimator_, X_test, y_test)
    results_all.append(xgb_result)
    joblib.dump(xgb_search.best_estimator_, os.path.join(MODELS_DIR, "xgboost.pkl"))

    # ── Select best model by AUC ──────────────────────────────────────────────
    best_result = max(results_all, key=lambda r: r["auc"])
    best_name   = best_result["model"]
    model_map   = {
        "Logistic Regression": lr_search.best_estimator_,
        "Random Forest":       rf_search.best_estimator_,
        "XGBoost":             xgb_search.best_estimator_,
    }
    best_model = model_map[best_name]

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    logger.info("\n✅  Best model: %s  (AUC=%.4f)", best_name, best_result["auc"])

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        "best_model":      best_name,
        "numeric_cols":    NUM_COLS,
        "categorical_cols": CAT_COLS,
        "passthrough_cols": PASS_COLS,
        "results":         results_all,
        "best_params":     str(xgb_search.best_params_),
    }
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Metadata saved → %s", meta_path)

    # ── Pretty summary ────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("   MODEL COMPARISON SUMMARY")
    print("═" * 52)
    print(f"  {'Model':<24} {'AUC':>6}  {'F1':>6}  {'Acc':>6}")
    print("─" * 52)
    for r in results_all:
        marker = " ◄ BEST" if r["model"] == best_name else ""
        print(f"  {r['model']:<24} {r['auc']:>6.4f}  {r['f1']:>6.4f}  {r['accuracy']:>6.4f}{marker}")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    main()