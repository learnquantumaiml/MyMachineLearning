import numpy as np                                  # for generating and manipulating numeric data
import pandas as pd                                 # for building a table (DataFrame) of features
from sklearn.model_selection import train_test_split   # splits data into "train" and "test" sets
from sklearn.compose import ColumnTransformer           # applies different preprocessing to different columns
from sklearn.pipeline import Pipeline                   # chains preprocessing + model into one object
from sklearn.impute import SimpleImputer                 # fills in missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # scale numbers / encode categories
from sklearn.linear_model import LogisticRegression       # the model we'll train (kept simple on purpose)


def make_churn_dataset(n_samples=800, random_state=42):
    """
    Builds a SYNTHETIC (fake, but realistic) telecom customer-churn dataset.
    We generate our own data instead of downloading a file so this script
    runs standalone for anyone who clones the repo — no dataset download needed.

    "Churn" means a customer cancels their subscription. We're simulating:
      - age              (number)
      - tenure_months    (number) -> how long they've been a customer
      - monthly_charges  (number) -> what they pay per month
      - contract_type    (category) -> Month-to-month / One year / Two year
      - churn            (0 or 1)  -> did they leave? This is what we predict.
    """
    rng = np.random.default_rng(random_state)  # a seeded random generator so results are reproducible

    # --- generate the raw feature columns -------------------------------
    age = rng.integers(18, 75, size=n_samples)                       # random ages between 18 and 74
    tenure_months = rng.integers(0, 72, size=n_samples)               # 0 to 71 months as a customer
    monthly_charges = rng.normal(70, 25, size=n_samples).clip(20, 150)  # bell-curve charges, clipped to a sane range
    contract_type = rng.choice(                                       # pick a contract type per customer
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.25, 0.20],  # most customers are month-to-month, like in real telecom data
    )

    # --- build a REAL underlying signal, so the model has something to learn ---
    # Flexible (month-to-month) contracts churn more; we encode that as a "risk" number.
    contract_risk = np.where(
        contract_type == "Month-to-month", 1.0,
        np.where(contract_type == "One year", 0.4, 0.1),
    )
    risk_score = (
        0.09 * (monthly_charges - 70)         # paying more than average -> higher churn risk
        - 0.12 * tenure_months                 # longer-tenured customers churn less
        + 5.0 * contract_risk                  # flexible contracts churn more
        + rng.normal(0, 0.3, size=n_samples)   # a little random noise, so it's not a 100%-predictable rule
    )
    churn_probability = 1 / (1 + np.exp(-risk_score / 1.2))  # squashes risk_score into a 0-to-1 probability
    churn = (rng.random(n_samples) < churn_probability).astype(int)  # roll the dice per customer using that probability

    # --- assemble everything into one table (DataFrame) -----------------
    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "contract_type": contract_type,
        "churn": churn,
    })

    # --- sprinkle in a few missing values, like real messy data would have ---
    missing_idx = rng.choice(n_samples, size=int(n_samples * 0.03), replace=False)  # pick ~3% of rows
    df.loc[missing_idx, "monthly_charges"] = np.nan  # blank out monthly_charges for those rows

    return df


def main():
    # ---- 1. Get the data -------------------------------------------------
    df = make_churn_dataset()
    X = df.drop(columns=["churn"])  # X = every column EXCEPT the answer (the standard ML naming convention)
    y = df["churn"]                  # y = the answer column we want the model to predict

    # 🔧 CHANGE ME: these lists tell the pipeline which columns need which treatment.
    # If you add a new numeric or categorical column to the dataset above, add its
    # name to the matching list below or the pipeline won't know what to do with it.
    numeric_features = ["age", "tenure_months", "monthly_charges"]
    categorical_features = ["contract_type"]

    # ---- 2. Define preprocessing for NUMERIC columns ----------------------
    # Two steps, always applied in this order:
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),  # fill missing numbers with the column's median (robust to outliers)
        ("scale", StandardScaler()),                     # rescale so every number has mean 0, std 1 (models like this)
    ])

    # ---- 3. Define preprocessing for CATEGORICAL columns -------------------
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),           # fill missing categories with the most common one
        ("onehot", OneHotEncoder(handle_unknown="ignore")),            # turn "Month-to-month"/"One year"/etc into 0/1 columns
        # handle_unknown="ignore" -> if a NEW category shows up later that the
        # model never saw during training, don't crash, just encode it as all-zeros.
        # This one setting is exactly the kind of "won't break in production"
        # detail AI-300 cares about.
    ])

    # ---- 4. Combine both into a single ColumnTransformer --------------------
    # This says: "run numeric_pipeline on numeric_features, categorical_pipeline
    # on categorical_features, then glue the results back together side by side."
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    # ---- 5. Chain preprocessing + model into ONE pipeline object -------------
    # 🔧 CHANGE ME: swap LogisticRegression for any other scikit-learn classifier
    # (e.g. RandomForestClassifier()) — the rest of this script doesn't need to change.
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classify", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # ---- 6. Split into train and test sets ------------------------------------
    # 🔧 CHANGE ME: test_size=0.2 means 20% of rows are held back to test on.
    # stratify=y keeps the churn/no-churn ratio the same in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- 7. Fit the WHOLE pipeline in one call ---------------------------------
    # This single .fit() call does everything: impute -> scale -> one-hot -> train.
    # No manual step-by-step transforming required — that's the point of Pipeline.
    model.fit(X_train, y_train)

    # ---- 8. Evaluate on the held-out test set -----------------------------------
    accuracy = model.score(X_test, y_test)  # fraction of test-set predictions that were correct

    # ---- 9. Show it working on a couple of brand-new, never-seen customers -------
    new_customers = pd.DataFrame({
        "age": [25, 60],
        "tenure_months": [2, 48],
        "monthly_charges": [95.0, 55.0],
        "contract_type": ["Month-to-month", "Two year"],
    })
    predictions = model.predict(new_customers)

    # ---- 10. Print results in plain language --------------------------------------
    print(f"Test accuracy: {accuracy:.2%}")
    print("(This means the pipeline correctly predicted churn/no-churn for "
          f"{accuracy:.0%} of customers it had never seen before.)\n")

    for i, pred in enumerate(predictions):
        verdict = "LIKELY TO CHURN" if pred == 1 else "likely to stay"
        print(f"New customer {i + 1}: {verdict}")


if __name__ == "__main__":
    main()
