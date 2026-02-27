import pandas as pd

file_path = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"

df = pd.read_csv(file_path, header=None, na_values=["?"])

print("Dataset shape:", df.shape)

conduction_cols = list(range(19, 26))

cond = df[conduction_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
cond = cond.clip(lower=0, upper=1).astype(int)

df["conduction_history_count"] = cond.sum(axis=1)
df["conduction_history_any"] = (df["conduction_history_count"] > 0).astype(int)

print("\nDistribution of conduction_history_count:")
print(df["conduction_history_count"].value_counts().sort_index())

positive_cases = df[df["conduction_history_count"] > 0]

print("\nNumber of positive cases:", len(positive_cases))

print("\nSample positive rows:")
print(
    positive_cases[
        conduction_cols +
        ["conduction_history_count", "conduction_history_any"]
    ].head(20)
)

print("\nManual verification of first 10 positive rows:")
for idx, row in positive_cases.head(10).iterrows():
    manual_sum = row[conduction_cols].sum()
    computed = row["conduction_history_count"]
    print(f"Row {idx}: manual={manual_sum}, computed={computed}")

print("\nAggregation complete.")