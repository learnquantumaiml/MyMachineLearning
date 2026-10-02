import pandas as pd

# Simulate raw customer-support ticket data before it becomes model-ready features
data = {
    "customer_id": [1, 2, 3, 4, 5, 6],
    "account_age_days": [45, 720, 15, 1500, 200, 3],
    "monthly_spend": [12.5, 89.0, 5.0, 150.0, 40.0, 2.0],
    "plan_type": ["free", "premium", "free", "enterprise", "premium", "free"],
    "num_support_tickets": [1, 0, 4, 0, 2, 6],
}
df = pd.DataFrame(data)
print("---- raw features ----")
print(df)

# pd.cut() slices a numeric column into buckets you define.
# bins are the cut points, labels are the names for the ranges BETWEEN those cut points,
# so there's always one fewer label than there are bin edges.
# change the numbers in "bins" if your own data needs different cutoffs
print("\n---- step 1: bin a numeric column into readable categories ----")
bins = [0, 30, 365, 100000]
labels = ["new_user", "established_user", "veteran_user"]
df["account_age_bucket"] = pd.cut(df["account_age_days"], bins=bins, labels=labels)
print(df[["account_age_days", "account_age_bucket"]])

# most ML algorithms can only work with numbers, not text labels like "free" or "premium".
# get_dummies() turns plan_type into separate 0/1 columns, one per category.
# the "columns" argument tells it which column(s) to encode this way
print("\n---- step 2: one-hot encode a categorical column ----")
df_encoded = pd.get_dummies(df, columns=["plan_type"], prefix="plan")
print(df_encoded)

# an engineered feature is just a new column built out of existing ones.
# here, spend divided by account age gives a "how much do they spend, adjusted
# for how long they've been a customer" signal that neither original column captured alone
print("\n---- step 3: create a new feature from existing columns ----")
df_encoded["spend_per_day"] = (df_encoded["monthly_spend"] / df_encoded["account_age_days"]).round(3)
print(df_encoded[["customer_id", "monthly_spend", "account_age_days", "spend_per_day"]])

# a rule-based feature: combine two conditions with & (AND) to flag a pattern
# you already believe matters, in this case, lots of tickets plus low spend.
# adjust the "3" and "20" thresholds below to match your own definition of risk
print("\n---- step 4: flag a simple risk feature ----")
df_encoded["high_churn_risk"] = ((df_encoded["num_support_tickets"] >= 3) & (df_encoded["monthly_spend"] < 20)).astype(int)
print(df_encoded[["customer_id", "num_support_tickets", "monthly_spend", "high_churn_risk"]])
