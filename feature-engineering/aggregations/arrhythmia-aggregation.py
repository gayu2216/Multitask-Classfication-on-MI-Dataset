import pandas as pd


file_path = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"

df = pd.read_csv(file_path, header=None, na_values=["?"])

print("Dataset shape:", df.shape)

arrhythmia_cols = list(range(12, 19))

arr = df[arrhythmia_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
arr = arr.clip(lower=0, upper=1).astype(int)

df["arrhythmia_history_count"] = arr.sum(axis=1)
df["arrhythmia_history_any"] = (df["arrhythmia_history_count"] > 0).astype(int)

print("\nDistribution of arrhythmia_history_count:")
print(df["arrhythmia_history_count"].value_counts().sort_index())

positive_cases = df[df["arrhythmia_history_count"] > 0]

print("\nNumber of positive cases:", len(positive_cases))

print("\nSample positive rows:")
print(
    positive_cases[
        arrhythmia_cols +
        ["arrhythmia_history_count", "arrhythmia_history_any"]
    ].head(20)
)

print("\nManual verification of first 10 positive rows:")
for idx, row in positive_cases.head(10).iterrows():
    manual_sum = row[arrhythmia_cols].sum()
    computed = row["arrhythmia_history_count"]
    print(f"Row {idx}: manual={manual_sum}, computed={computed}")

output_path = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI_step1.csv"
df.to_csv(output_path, index=False)

print("\nSaved engineered dataset to:", output_path)