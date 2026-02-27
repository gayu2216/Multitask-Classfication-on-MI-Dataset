import pandas as pd

file_path = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"

df = pd.read_csv(file_path, header=None, na_values=["?"])

print("Dataset shape:", df.shape)

lung_cols = list(range(29, 34))

lung = df[lung_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
lung = lung.clip(lower=0, upper=1).astype(int)

df["lung_disease_count"] = lung.sum(axis=1)
df["lung_disease_any"] = (df["lung_disease_count"] > 0).astype(int)

print("\nDistribution of lung_disease_count:")
print(df["lung_disease_count"].value_counts().sort_index())

positive_cases = df[df["lung_disease_count"] > 0]

print("\nNumber of positive cases:", len(positive_cases))

print("\nSample positive rows:")
print(
    positive_cases[
        lung_cols +
        ["lung_disease_count", "lung_disease_any"]
    ].head(20)
)

print("\nManual verification of first 10 positive rows:")
for idx, row in positive_cases.head(10).iterrows():
    manual_sum = row[lung_cols].sum()
    computed = row["lung_disease_count"]
    print(f"Row {idx}: manual={manual_sum}, computed={computed}")

print("\nAggregation complete.")