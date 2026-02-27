import pandas as pd

file_path = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"

df = pd.read_csv(file_path, header=None, na_values=["?"])

print("Dataset shape:", df.shape)

ecg_conduction_cols = list(range(64, 75))

ecg = df[ecg_conduction_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
ecg = ecg.clip(lower=0, upper=1).astype(int)

df["ecg_conduction_count"] = ecg.sum(axis=1)
df["ecg_conduction_any"] = (df["ecg_conduction_count"] > 0).astype(int)

print("\nDistribution of ecg_conduction_count:")
print(df["ecg_conduction_count"].value_counts().sort_index())

positive_cases = df[df["ecg_conduction_count"] > 0]

print("\nNumber of patients with conduction abnormality:", len(positive_cases))

print("\nSample positive rows:")
print(
    positive_cases[
        ecg_conduction_cols +
        ["ecg_conduction_count", "ecg_conduction_any"]
    ].head(20)
)

print("\nManual verification of first 10 positive rows:")
for idx, row in positive_cases.head(10).iterrows():
    manual_count = row[ecg_conduction_cols].sum()
    print(
        f"Row {idx}: manual={manual_count}, computed={row['ecg_conduction_count']}"
    )

print("\nECG conduction aggregation complete.")