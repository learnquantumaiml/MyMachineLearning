import numpy as np                                              # numeric operations
from sklearn.model_selection import train_test_split               # splits data into train/test
from sklearn.linear_model import LinearRegression, Ridge             # the two models we'll compare
from sklearn.metrics import (
    mean_absolute_error,     # MAE: average size of the error, in the original units (e.g. dollars)
    mean_squared_error,       # used to derive RMSE below
    r2_score,                  # R²: what fraction of the variation in price our model explains
)


def make_rental_dataset(n_samples=500, random_state=42):
    """
    Builds a SYNTHETIC apartment-rental dataset so this script needs no
    external file. Features:
      - size_sqm            (bigger apartment -> more rent)
      - bedrooms             (more bedrooms -> more rent)
      - distance_km          (further from city center -> less rent)
      - building_age_years   (older building -> slightly less rent)
      - monthly_rent         (the number we want to predict)
    """
    rng = np.random.default_rng(random_state)  # seeded random generator, for reproducible results

    size_sqm = rng.normal(75, 25, size=n_samples).clip(20, 200)         # apartment size in square meters
    bedrooms = rng.integers(1, 5, size=n_samples)                        # 1 to 4 bedrooms
    distance_km = rng.exponential(scale=5, size=n_samples).clip(0.2, 30)  # most apartments cluster near center
    building_age_years = rng.integers(0, 60, size=n_samples)              # how old the building is

    # 🔧 CHANGE ME: these coefficients control how much each feature affects
    # rent. Try changing them to see how the model's learned weights (printed
    # below) change to match whatever pattern you build into the data.
    monthly_rent = (
        400                                  # base rent everyone pays, regardless of anything else
        + 12.0 * size_sqm                    # each extra sqm adds ~12 currency units
        + 80.0 * bedrooms                    # each extra bedroom adds ~80
        - 15.0 * distance_km                 # each km from the center subtracts ~15
        - 2.0 * building_age_years           # each year of building age subtracts ~2
        + rng.normal(0, 60, size=n_samples)  # random noise (things our features don't capture)
    ).clip(250, None)  # rent can't go below a sane floor

    X = np.column_stack([size_sqm, bedrooms, distance_km, building_age_years])
    y = monthly_rent
    feature_names = ["size_sqm", "bedrooms", "distance_km", "building_age_years"]
    return X, y, feature_names


def report_metrics(name, y_true, y_pred):
    """Prints MAE, RMSE, and R² for one model's predictions, in plain language."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # RMSE = square root of average squared error
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"MAE  : {mae:8.2f}  (on average, predictions are off by this many currency units)")
    print(f"RMSE : {rmse:8.2f}  (like MAE, but punishes big misses harder than small ones)")
    print(f"R²   : {r2:8.3f}  (fraction of rent variation explained; 1.0 = perfect, 0.0 = no better than guessing the average)\n")
    return mae, rmse, r2


def main():
    # ---- 1. Get the data --------------------------------------------------
    X, y, feature_names = make_rental_dataset()

    # ---- 2. Split into train and test sets -----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- 3. Train a plain Linear Regression -----------------------------------
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred_linreg = linreg.predict(X_test)

    # ---- 4. Train a Ridge Regression (same idea, plus a regularization penalty) --
    # 🔧 CHANGE ME: alpha controls how strongly the model is penalized for large
    # coefficients. alpha=0 behaves like plain LinearRegression; higher alpha =
    # a simpler, more conservative model that's less likely to overfit noisy data.
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # ---- 5. Report metrics for both models -------------------------------------
    print("=" * 65)
    print("Comparing a plain model vs a regularized model")
    print("=" * 65)
    report_metrics("Linear Regression (no penalty)", y_test, y_pred_linreg)
    report_metrics("Ridge Regression (alpha=10.0 penalty)", y_test, y_pred_ridge)

    # ---- 6. Show what each model actually learned, feature by feature -------------
    print("=" * 65)
    print("Coefficients learned: plain vs regularized (Ridge nudges these toward 0)")
    print("=" * 65)
    for name, coef_lr, coef_rd in zip(feature_names, linreg.coef_, ridge.coef_):
        shrink_pct = (1 - abs(coef_rd) / abs(coef_lr)) * 100  # how much smaller Ridge's version is, in %
        print(f"  {name:20s}: LinearRegression={coef_lr:8.2f}   Ridge={coef_rd:8.2f}   (shrunk {shrink_pct:5.2f}%)")

    # ---- 7. Predict rent for one brand-new apartment ------------------------------
    new_apartment = np.array([[90, 3, 2.5, 10]])  # 90 sqm, 3 bedrooms, 2.5km out, 10 years old
    predicted_rent = linreg.predict(new_apartment)[0]
    print(f"\nPredicted rent for a 90sqm, 3-bedroom apartment 2.5km out, 10 years old: "
          f"{predicted_rent:.2f}")


if __name__ == "__main__":
    main()
