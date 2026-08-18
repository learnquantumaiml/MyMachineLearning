import numpy as np
import pandas as pd
from scipy.stats import ks_2samp                                # Kolmogorov-Smirnov test: compares two numeric distributions

DRIFT_P_VALUE_THRESHOLD = 0.05

CATEGORY_DRIFT_THRESHOLD = 0.10  # 10 percentage points

def make_churn_dataset(n_samples=800, random_state=42, price_shift=0.0, contract_shift=False):
    """
    Same churn data generator as Day 1, but with two optional "what changed in
    the real world" knobs:
      - price_shift    : adds this amount to every customer's monthly_charges
                          (simulates a company-wide price increase)
      - contract_shift  : if True, skews more customers toward "Month-to-month"
                          (simulates a promotion, or an economic downturn making
                          customers avoid long commitments)
    """
    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 75, size=n_samples)
    tenure_months = rng.integers(0, 72, size=n_samples)
    monthly_charges = (rng.normal(70, 25, size=n_samples) + price_shift).clip(20, 220)

    contract_probs = [0.75, 0.15, 0.10] if contract_shift else [0.55, 0.25, 0.20]
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=contract_probs)

    contract_risk = np.where(contract_type == "Month-to-month", 1.0, np.where(contract_type == "One year", 0.4, 0.1))
    risk_score = (
        0.09 * (monthly_charges - 70) - 0.12 * tenure_months + 5.0 * contract_risk
        + rng.normal(0, 0.3, size=n_samples)
    )
    churn_probability = 1 / (1 + np.exp(-risk_score / 1.2))
    churn = (rng.random(n_samples) < churn_probability).astype(int)

    return pd.DataFrame({
        "age": age, "tenure_months": tenure_months, "monthly_charges": monthly_charges,
        "contract_type": contract_type, "churn": churn,
    })


def check_numeric_drift(reference: pd.Series, current: pd.Series, feature_name: str):
    """Runs a Kolmogorov-Smirnov test comparing two numeric columns and reports the verdict."""
    ks_statistic, p_value = ks_2samp(reference, current)   # ks_statistic: how different the two distributions are (0=identical)
    drifted = p_value < DRIFT_P_VALUE_THRESHOLD              # small p-value = "this difference is very unlikely to be chance"
    verdict = "🚨 DRIFT DETECTED" if drifted else "OK, no significant drift"
    print(f"  {feature_name:20s} | KS stat={ks_statistic:.3f}  p-value={p_value:.4f}  -> {verdict}")
    print(f"    (reference mean={reference.mean():.2f}, current mean={current.mean():.2f})")
    return drifted


def check_categorical_drift(reference: pd.Series, current: pd.Series, feature_name: str):
    """Compares category proportions between two batches and flags any big shifts."""
    ref_props = reference.value_counts(normalize=True)   # normalize=True -> gives proportions (0-1), not raw counts
    cur_props = current.value_counts(normalize=True)
    drifted = False
    print(f"  {feature_name:20s}")
    for category in ref_props.index.union(cur_props.index):   # union covers categories present in either batch
        ref_pct = ref_props.get(category, 0.0)
        cur_pct = cur_props.get(category, 0.0)
        shift = abs(cur_pct - ref_pct)
        if shift > CATEGORY_DRIFT_THRESHOLD:
            drifted = True
        flag = " 🚨" if shift > CATEGORY_DRIFT_THRESHOLD else ""
        print(f"    {category:20s}: reference={ref_pct:.1%}  current={cur_pct:.1%}  shift={shift:.1%}{flag}")
    verdict = "🚨 DRIFT DETECTED" if drifted else "OK, no significant drift"
    print(f"    -> {verdict}")
    return drifted


def main():
    # ---- 1. "Reference" data = what the model was originally trained on ---------
    reference_df = make_churn_dataset(random_state=42)

    # ---- 2. "Current" data = a new batch collected after deployment --------------
    # 🔧 CHANGE ME: try price_shift=0 and contract_shift=False to see what a
    # NO-drift report looks like (everything should come back "OK" below).
    current_df = make_churn_dataset(
        n_samples=500,
        random_state=99,       # a different seed = a genuinely new random batch
        price_shift=25.0,       # simulate a price increase of 25 currency units
        contract_shift=True,    # simulate a promotion pushing more month-to-month signups
    )

    print("=" * 65)
    print("NUMERIC FEATURE DRIFT (Kolmogorov-Smirnov test)")
    print("=" * 65)
    numeric_features = ["age", "tenure_months", "monthly_charges"]
    numeric_drift_flags = [
        check_numeric_drift(reference_df[col], current_df[col], col) for col in numeric_features
    ]

    print("\n" + "=" * 65)
    print("CATEGORICAL FEATURE DRIFT (proportion comparison)")
    print("=" * 65)
    categorical_drift_flags = [
        check_categorical_drift(reference_df["contract_type"], current_df["contract_type"], "contract_type")
    ]

    # ---- 3. Overall verdict ------------------------------------------------------
    any_drift = any(numeric_drift_flags) or any(categorical_drift_flags)
    print("\n" + "=" * 65)
    if any_drift:
        print("OVERALL: 🚨 Drift detected in at least one feature.")
        print("This does NOT necessarily mean the model is broken — it means the")
        print("world changed and the model should be RE-EVALUATED (and probably")
        print("retrained) soon, before its predictions quietly get worse.")
    else:
        print("OVERALL: ✅ No significant drift detected. Model is likely still")
        print("seeing the kind of data it was trained on.")
    print("=" * 65)


if __name__ == "__main__":
    main()
