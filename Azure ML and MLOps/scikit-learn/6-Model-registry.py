import json                                                       # for writing/reading the metadata file
import datetime as dt                                               # to timestamp when the model was trained
from pathlib import Path                                              # for clean, cross-platform file paths
import numpy as np
import pandas as pd
import joblib                                                          # the library that saves/loads scikit-learn objects
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

REGISTRY_DIR = Path("model_registry")

def make_churn_dataset(n_samples=800, random_state=42):
    """Same synthetic churn dataset as Day 1 / Day 5 — copied here so this file runs standalone."""
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 75, size=n_samples)
    tenure_months = rng.integers(0, 72, size=n_samples)
    monthly_charges = rng.normal(70, 25, size=n_samples).clip(20, 150)
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20]
    )
    contract_risk = np.where(
        contract_type == "Month-to-month", 1.0, np.where(contract_type == "One year", 0.4, 0.1)
    )
    risk_score = (
        0.09 * (monthly_charges - 70) - 0.12 * tenure_months + 5.0 * contract_risk
        + rng.normal(0, 0.3, size=n_samples)
    )
    churn_probability = 1 / (1 + np.exp(-risk_score / 1.2))
    churn = (rng.random(n_samples) < churn_probability).astype(int)
    df = pd.DataFrame({
        "age": age, "tenure_months": tenure_months, "monthly_charges": monthly_charges,
        "contract_type": contract_type, "churn": churn,
    })
    missing_idx = rng.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_idx, "monthly_charges"] = np.nan
    return df


def build_pipeline():
    """Same Pipeline structure as Day 1 / Day 5."""
    numeric_features = ["age", "tenure_months", "monthly_charges"]
    categorical_features = ["contract_type"]
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return Pipeline([("preprocess", preprocessor), ("classify", LogisticRegression(max_iter=1000, random_state=42))])


def train_and_register(version: str):
    """Trains a fresh pipeline, saves it to disk, and writes a metadata JSON beside it."""
    df = make_churn_dataset()
    X, y = df.drop(columns=["churn"]), df["churn"]

    # cross-validate first, so the metadata records an HONEST performance estimate
    # (not just a single lucky split — that's the Day 5 lesson, reused here)
    cv_scores = cross_val_score(
        build_pipeline(), X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy"
    )

    # now fit the FINAL model on ALL the data (no need to hold back a test set —
    # cross-validation above already gave us an honest performance estimate)
    final_model = build_pipeline()
    final_model.fit(X, y)

    # ---- save the model file itself --------------------------------------------
    REGISTRY_DIR.mkdir(exist_ok=True)                        # create the folder if it doesn't exist yet
    model_path = REGISTRY_DIR / f"churn_model_{version}.joblib"
    joblib.dump(final_model, model_path)                       # this is the ONE line that actually saves the model

    # ---- save metadata alongside it, the way a model registry would --------------
    metadata = {
        "version": version,
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "algorithm": "LogisticRegression (inside a preprocessing Pipeline)",
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "n_training_rows": len(df),
        "expected_input_columns": list(X.columns),
        "target_column": "churn",
    }
    metadata_path = REGISTRY_DIR / f"churn_model_{version}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)  # indent=2 makes the file human-readable, not just machine-readable

    return model_path, metadata_path, metadata


def main():
    # ---- 1. Train a model and "register" it (save model + metadata) --------------
    # 🔧 CHANGE ME: bump this version string every time you retrain, the same way
    # you'd version a model in an Azure ML model registry (v1, v2, v3, ...).
    version = "v1"
    model_path, metadata_path, metadata = train_and_register(version)

    print("=" * 65)
    print(f"Saved model to:    {model_path}")
    print(f"Saved metadata to: {metadata_path}")
    print("=" * 65)
    print(json.dumps(metadata, indent=2))

    # ---- 2. Simulate a COMPLETELY SEPARATE program loading the model later ---------
    # In real life this next block would be a different file entirely — an API
    # server, a batch scoring job, a scheduled task. We put it in the same file
    # here purely so the whole demo runs top-to-bottom with one command.
    print("\n" + "=" * 65)
    print("Simulating a separate program loading the saved model...")
    print("=" * 65)
    reloaded_model = joblib.load(model_path)   # this is the ONE line that loads a saved model back into memory

    new_customers = pd.DataFrame({
        "age": [25, 60],
        "tenure_months": [2, 48],
        "monthly_charges": [95.0, 55.0],
        "contract_type": ["Month-to-month", "Two year"],
    })
    predictions = reloaded_model.predict(new_customers)
    for i, pred in enumerate(predictions):
        verdict = "LIKELY TO CHURN" if pred == 1 else "likely to stay"
        print(f"Reloaded model — new customer {i + 1}: {verdict}")

    print("\nThe reloaded model made predictions with ZERO retraining — it's the")
    print("exact same fitted pipeline, just read back from disk. That's the whole")
    print("point: train expensively ONCE, reuse cheaply as many times as you need.")


if __name__ == "__main__":
    main()
