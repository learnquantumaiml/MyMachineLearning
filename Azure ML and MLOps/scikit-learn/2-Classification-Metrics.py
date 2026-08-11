import numpy as np                                          # numeric operations
from sklearn.datasets import make_classification              # generates a synthetic classification dataset
from sklearn.model_selection import train_test_split           # splits data into train/test
from sklearn.ensemble import RandomForestClassifier              # the model we'll train
from sklearn.dummy import DummyClassifier                        # a "does-nothing-smart" baseline model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,   # the core metrics
    confusion_matrix, roc_auc_score, classification_report,     # more detailed reporting tools
)


def main():
    # ---- 1. Build a synthetic, IMBALANCED loan-default dataset -------------
    # 🔧 CHANGE ME: weights=[0.75, 0.25] means ~75% "no default", ~25% "default".
    # Real-world default/fraud/churn/failure datasets are almost always this
    # imbalanced — that imbalance is exactly why accuracy alone lies to you.
    X, y = make_classification(
        n_samples=1200,        # 🔧 CHANGE ME: total number of loan applicants to simulate
        n_features=10,         # total number of input columns
        n_informative=6,       # how many of those columns actually carry signal
        n_redundant=2,         # extra columns that are combinations of the informative ones
        weights=[0.75, 0.25],  # class balance: 75% class 0 (no default), 25% class 1 (default)
        flip_y=0.03,           # a small % of intentionally mislabeled rows, like real noisy data
        random_state=42,       # fixes the randomness so results are reproducible
    )

    # ---- 2. Split into train and test sets ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y  # stratify keeps the 75/25 split even in both halves
    )

    # ---- 3. Train the REAL model --------------------------------------------
    # 🔧 CHANGE ME: n_estimators = number of decision trees in the forest.
    # More trees = usually better, but slower. class_weight="balanced" tells
    # the model to pay extra attention to the rare "default" class instead of
    # ignoring it to chase raw accuracy.
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)                    # hard predictions: 0 or 1
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # soft predictions: probability of class 1 (default)

    # ---- 4. Train a LAZY baseline for comparison -----------------------------
    # This model does zero learning — it just always predicts the most common class.
    # We train it purely to prove a point about accuracy in step 6 below.
    baseline = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    # ---- 5. Compute every metric ------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    precision = precision_score(y_test, y_pred)   # of everyone we FLAGGED as default, how many really defaulted?
    recall = recall_score(y_test, y_pred)          # of everyone who REALLY defaulted, how many did we catch?
    f1 = f1_score(y_test, y_pred)                  # a single score balancing precision and recall
    roc_auc = roc_auc_score(y_test, y_pred_proba)   # how well the model ranks "risky" applicants above "safe" ones
    cm = confusion_matrix(y_test, y_pred)             # the 2x2 table of right/wrong predictions by type

    # ---- 6. Print everything in plain language -----------------------------------
    print("=" * 65)
    print("STEP 1: Why accuracy alone is misleading on imbalanced data")
    print("=" * 65)
    print(f"Lazy baseline (always predicts 'no default') accuracy: {acc_baseline:.2%}")
    print("^ That number looks great, but this model NEVER catches a single")
    print("  defaulter. It's a useless model wearing a good accuracy score.\n")

    print("=" * 65)
    print("STEP 2: Our real model's numbers")
    print("=" * 65)
    print(f"Accuracy : {acc:.2%}   (overall, how often are we right?)")
    print(f"Precision: {precision:.2%}   (of applicants we flagged as risky, this % really default)")
    print(f"Recall   : {recall:.2%}   (of applicants who really default, we catch this %)")
    print(f"F1 score : {f1:.2%}   (balances precision and recall into one number)")
    print(f"ROC-AUC  : {roc_auc:.2%}   (how well we RANK risky vs safe applicants, 50%=random guessing)\n")

    print("=" * 65)
    print("STEP 3: Confusion matrix — where exactly the mistakes happen")
    print("=" * 65)
    tn, fp, fn, tp = cm.ravel()
    print(f"True negatives  (correctly said 'safe'):    {tn}")
    print(f"False positives (wrongly flagged as risky): {fp}   <- annoys good customers")
    print(f"False negatives (missed a real defaulter):  {fn}   <- costs the bank money")
    print(f"True positives  (correctly caught risky):   {tp}\n")

    print("=" * 65)
    print("STEP 4: scikit-learn's built-in summary (same numbers, one table)")
    print("=" * 65)
    print(classification_report(y_test, y_pred, target_names=["no_default", "default"]))

    print("WHICH METRIC MATTERS MOST? It depends on the business cost:")
    print("- Missing a real defaulter (false negative) costs the bank real money")
    print("  -> in that case, RECALL matters more than precision.")
    print("- Wrongly flagging good customers (false positive) annoys/loses them")
    print("  -> in that case, PRECISION matters more than recall.")
    print("There is no single 'best' metric — you pick based on which mistake hurts more.")


if __name__ == "__main__":
    main()
