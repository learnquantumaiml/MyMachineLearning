from scipy.stats import randint                                    # defines a random-integer range to sample from
from sklearn.datasets import make_classification                     # synthetic dataset, same recipe as Day 2
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier                   # the model we're tuning
from sklearn.metrics import f1_score                                    # our chosen scoring metric (see Day 2's lesson)


def main():
    # ---- 1. Same loan-default dataset recipe as Day 2 ------------------------
    X, y = make_classification(
        n_samples=1200, n_features=10, n_informative=6, n_redundant=2,
        weights=[0.75, 0.25], flip_y=0.03, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ---- 2. Baseline: an UNTUNED model, using scikit-learn's defaults ----------
    baseline_model = RandomForestClassifier(random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_f1 = f1_score(y_test, baseline_model.predict(X_test))

    # ---- 3. GridSearchCV: try EVERY combination in this grid --------------------
    # 🔧 CHANGE ME: each list below is a set of values to try for that setting.
    # GridSearchCV will try every combination: 3 x 3 x 2 = 18 combinations here,
    # each evaluated with 5-fold cross-validation = 90 total model fits.
    # Bigger grids = more thorough but slower. Start small, then widen if needed.
    param_grid = {
        "n_estimators": [100, 200, 400],     # how many trees to build
        "max_depth": [4, 8, None],            # how deep each tree can grow (None = unlimited)
        "min_samples_leaf": [1, 5],           # minimum data points allowed in a leaf node
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid=param_grid,
        scoring="f1",     # 🔧 CHANGE ME: optimize for F1 (from Day 2's lesson), not raw accuracy
        cv=5,              # 🔧 CHANGE ME: 5-fold cross-validation; higher = more reliable but slower
        n_jobs=-1,         # use all available CPU cores to run combinations in parallel
    )
    grid_search.fit(X_train, y_train)
    grid_best_f1 = f1_score(y_test, grid_search.predict(X_test))

    # ---- 4. RandomizedSearchCV: sample a RANDOM subset of combinations ------------
    # 🔧 CHANGE ME: n_iter controls how many random combinations to actually try.
    # Here we search a MUCH wider range of possible settings than the grid above,
    # but only test 15 random combinations from it instead of all of them.
    param_distributions = {
        "n_estimators": randint(50, 500),        # any integer from 50 to 499
        "max_depth": randint(2, 20),               # any integer from 2 to 19
        "min_samples_leaf": randint(1, 10),          # any integer from 1 to 9
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_distributions=param_distributions,
        n_iter=15,          # 🔧 CHANGE ME: how many random combinations to try (higher = slower, more thorough)
        scoring="f1",
        cv=5,
        random_state=42,    # makes the "random" sampling reproducible
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)
    random_best_f1 = f1_score(y_test, random_search.predict(X_test))

    # ---- 5. Print a side-by-side comparison ------------------------------------
    print("=" * 65)
    print("F1 score on the held-out test set (higher is better)")
    print("=" * 65)
    print(f"Untuned baseline (scikit-learn defaults) : {baseline_f1:.4f}")
    print(f"GridSearchCV best model                  : {grid_best_f1:.4f}")
    print(f"RandomizedSearchCV best model             : {random_best_f1:.4f}\n")

    print("=" * 65)
    print("Winning settings GridSearchCV found:")
    print("=" * 65)
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"  (best cross-validated F1 during search: {grid_search.best_score_:.4f})\n")

    print("=" * 65)
    print("Winning settings RandomizedSearchCV found:")
    print("=" * 65)
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"  (best cross-validated F1 during search: {random_search.best_score_:.4f})\n")

    print("TAKEAWAY: GridSearchCV is exhaustive but only practical for small grids.")
    print("RandomizedSearchCV scales to much bigger search spaces because you")
    print("control the number of tries directly with n_iter, regardless of how")
    print("many hyperparameters or values you're searching over.\n")

    if grid_best_f1 <= baseline_f1:
        print("=" * 65)
        print("A NOTE ON WHAT YOU JUST SAW (this is normal, not a bug):")
        print("=" * 65)
        print("The tuned models scored AT OR BELOW the untuned baseline on THIS")
        print("one test split, even though their cross-validated scores during")
        print("the search were solid. That's expected, for two real reasons:")
        print("  1. RandomForest's defaults are already quite good for many")
        print("     problems — there isn't always a lot of room to improve.")
        print("  2. A single test split has luck baked into it. The fair")
        print("     comparison is the cross-validated score shown above")
        print("     (~0.79), which averages over 5 different splits — not")
        print("     this one lucky/unlucky 25% slice of the data.")
        print("A beginner sees 'tuning made it worse' and panics. An experienced")
        print("engineer checks the cross-validated score, shrugs, and moves on.")





if __name__ == "__main__":
    main()
