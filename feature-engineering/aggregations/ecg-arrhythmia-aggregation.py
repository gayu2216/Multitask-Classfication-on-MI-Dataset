import pandas as pd

file_path = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"


df = pd.read_csv(file_path, header=None, na_values=["?"])
print("Dataset shape:", df.shape)

ecg_arr_cols = list(range(55, 64))

ecg = df[ecg_arr_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
ecg = ecg.clip(lower=0, upper=1).astype(int)

df["ecg_arrhythmia_count"] = ecg.sum(axis=1)
df["ecg_arrhythmia_any"] = (df["ecg_arrhythmia_count"] > 0).astype(int)

df["ventricular_instability_any"] = (ecg.iloc[:, 7] + ecg.iloc[:, 8] > 0).astype(int)

df["atrial_instability_any"] = (ecg.iloc[:, 4] + ecg.iloc[:, 5] > 0).astype(int)

print("\nDistribution of ecg_arrhythmia_count:")
print(df["ecg_arrhythmia_count"].value_counts().sort_index())

print("\nventricular_instability_any distribution:")
print(df["ventricular_instability_any"].value_counts())

print("\natrial_instability_any distribution:")
print(df["atrial_instability_any"].value_counts())

positive_cases = df[df["ecg_arrhythmia_any"] > 0]
print("\nNumber of patients with any ECG arrhythmia:", len(positive_cases))

print("\nSample positive rows (ECG cols 56–64 + engineered):")
print(
    positive_cases[
        ecg_arr_cols +
        ["ecg_arrhythmia_count", "ecg_arrhythmia_any",
         "ventricular_instability_any", "atrial_instability_any"]
    ].head(25)
)

print("\nManual verification of first 10 positive rows:")
for idx, row in positive_cases.head(10).iterrows():
    manual = row[ecg_arr_cols].sum()
    computed = row["ecg_arrhythmia_count"]
    print(f"Row {idx}: manual={manual}, computed={computed}")

print("\nECG arrhythmia aggregation complete.")