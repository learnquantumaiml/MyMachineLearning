import pandas as pd

# a small batch of what real Azure OpenAI usage logs or MLflow run payloads
# often look like: nested dictionaries inside a list
raw_logs = [
    {
        "run_id": "run_101",
        "model": {"name": "gpt-4o-mini", "version": "2026-03-01"},
        "usage": {"prompt_tokens": 210, "completion_tokens": 85, "total_tokens": 295},
        "status": "succeeded",
    },
    {
        "run_id": "run_102",
        "model": {"name": "gpt-4o-mini", "version": "2026-03-01"},
        "usage": {"prompt_tokens": 340, "completion_tokens": 120, "total_tokens": 460},
        "status": "succeeded",
    },
    {
        "run_id": "run_103",
        "model": {"name": "phi-4", "version": "2026-01-15"},
        "usage": {"prompt_tokens": 150, "completion_tokens": 40, "total_tokens": 190},
        "status": "failed",
    },
]

print("---- this is what nested JSON looks like as a plain Python list ----")
print(raw_logs[0])
# notice "model" and "usage" are dictionaries INSIDE the dictionary, that's the nesting

# json_normalize flattens those inner dictionaries into their own columns.
# sep="." controls how the new column names are built, model["name"] becomes "model.name"
print("\n---- json_normalize flattens the nested keys into flat columns ----")
df = pd.json_normalize(raw_logs, sep=".")
print(df)

# once it's flat, it behaves exactly like every DataFrame from the last four days
print("\n---- now it behaves like any other DataFrame ----")
print("\ntotal tokens used per model:")
print(df.groupby("model.name")["usage.total_tokens"].sum())

print("\nfailed runs only:")
print(df[df["status"] == "failed"])

# the dotted column names work fine, but they're a little awkward to type
# repeatedly, so renaming them to something cleaner is a common last step
print("\n---- renaming the flattened columns to something cleaner ----")
df = df.rename(columns={
    "model.name": "model_name",
    "model.version": "model_version",
    "usage.prompt_tokens": "prompt_tokens",
    "usage.completion_tokens": "completion_tokens",
    "usage.total_tokens": "total_tokens",
})
print(df)
