import pandas as pd
import numpy as np

# Simulate a raw dataset of loan applications pulled in for a credit-risk model.
# Every row below was written to demonstrate ONE specific data quality problem,
# so it's easier to see which check catches which issue.
data = {
    "application_id": [101, 102, 103, 104, 104, 105],
    "applicant_age":   [34, 29, "forty-one", 45, 45, 190],   # row index 2 = wrong type, index 5 = out of range
    "annual_income":   [52000, 61000, 58000, np.nan, np.nan, 73000],  # index 3 and 4 = missing income
    "loan_amount":     [15000, 20000, 18000, 22000, 22000, 25000],
    "region":          ["EU", "EU", "APAC", "NA", "NA", "NA"],
}
df = pd.DataFrame(data)
# rows at index 3 and 4 are fully identical on purpose, to demonstrate a duplicate row

print("---- raw data ----")
print(df)

# isnull() returns True/False for every cell, sum() adds those up per column
# so you get a count of missing values, column by column
print("\n---- step 1: how many missing values in each column ----")
print(df.isnull().sum())

# duplicated() flags any row that is an exact copy of an earlier row
# drop_duplicates() removes them, keeping the first occurrence by default
print("\n---- step 2: find and remove duplicate rows ----")
print("duplicate rows found:", df.duplicated().sum())
df_clean = df.drop_duplicates().copy()  # .copy() avoids a pandas warning on the next step
print(df_clean)

# applicant_age has the string "forty-one" mixed in with numbers, so pandas
# treats the WHOLE column as text (object dtype), not numbers.
# pd.to_numeric with errors="coerce" converts what it can and turns anything
# it can't understand (like "forty-one") into NaN, instead of crashing.
print("\n---- step 3: fix the wrong data type on applicant_age ----")
df_clean["applicant_age"] = pd.to_numeric(df_clean["applicant_age"], errors="coerce")
print(df_clean)
print(df_clean.dtypes)  # confirm applicant_age is now float64, not object

# a value being a number doesn't mean it's a BELIEVABLE number.
# an age of 190 passed the "is this a number" check but still needs a sanity check.
# change the 18 and 100 below if your own dataset has a different valid age range
print("\n---- step 4: flag rows outside a believable age range ----")
bad_age_mask = (df_clean["applicant_age"] < 18) | (df_clean["applicant_age"] > 100) | (df_clean["applicant_age"].isnull())
print(df_clean[bad_age_mask])

# a quick summary table, the kind of thing you'd log or alert on
# before letting this data anywhere near a training job
print("\n---- step 5: one-glance data quality summary ----")
summary = pd.DataFrame({
    "missing_values": df_clean.isnull().sum(),
    "pct_missing": (df_clean.isnull().sum() / len(df_clean) * 100).round(1)
})
print(summary)
