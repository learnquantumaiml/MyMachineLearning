import numpy as np                                             # numeric operations
import pandas as pd                                              # building the churn dataset table
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer                      # per-column preprocessing, same as Day 1
from sklearn.pipeline import Pipeline                                # chains preprocessing + model
from sklearn.impute import SimpleImputer                              # fills missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder         # scale numbers / encode categories
from sklearn.linear_model import LogisticRegression                       # the model, same as Day 1


def make_churn_dataset(n_samples=800, random_state=42):
    """Same synthetic churn dataset as Day 1 — copied here so this file also runs standalone."""
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
        0.09 * (monthly_charges - 70)
        - 0.12 * tenure_months
        + 5.0 * contract_risk
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
    """Builds the exact same preprocessing + model Pipeline as Day 1, as a reusable function."""
    numeric_features = ["age", "tenure_months", "monthly_charges"]
    categorical_features = ["contract_type"]

    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classify", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def main():
    # ---- 1. Get the data ----------------------------------------------------
    df = make_churn_dataset()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # ---- 2. The OLD way: one single train/test split (like Day 1) -------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    single_split_model = build_pipeline()
    single_split_model.fit(X_train, y_train)
    single_split_score = single_split_model.score(X_test, y_test)

    # ---- 3. The BETTER way: 5-fold cross-validation on the whole pipeline -------
    # 🔧 CHANGE ME: n_splits=5 is the most common choice. More folds = more
    # reliable estimate but slower; fewer folds = faster but noisier estimate.
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # cross_val_score re-builds and re-fits the WHOLE pipeline (imputer, scaler,
    # one-hot encoder, model — everything) fresh on each fold's training data.
    # This is what correctly prevents test-fold information from leaking in.
    pipeline_for_cv = build_pipeline()
    cv_scores = cross_val_score(pipeline_for_cv, X, y, cv=cv_strategy, scoring="accuracy")

    # ---- 4. Print a plain-language comparison -----------------------------------
    print("=" * 65)
    print("Old way: score from ONE lucky (or unlucky) train/test split")
    print("=" * 65)
    print(f"Accuracy: {single_split_score:.2%}\n")

    print("=" * 65)
    print("Better way: 5-fold cross-validation on the SAME pipeline")
    print("=" * 65)
    for fold_number, score in enumerate(cv_scores, start=1):
        print(f"  Fold {fold_number}: {score:.2%}")
    print(f"\nMean accuracy across folds : {cv_scores.mean():.2%}")
    print(f"Standard deviation         : {cv_scores.std():.2%}")
    print("(A small standard deviation means the model performs consistently")
    print(" no matter which slice of data it sees — that's what makes the")
    print(" mean score trustworthy enough to report to a stakeholder.)")


if __name__ == "__main__":
    main()
