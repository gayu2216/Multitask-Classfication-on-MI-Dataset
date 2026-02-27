import pandas as pd

INPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"
OUTPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI_age_x_electrical.csv"

AGE_INDEX = 1
ECG_ARR_COLS = list(range(55, 64))  # 55..63

# --- load headerless CSV (comma or semicolon) ---
try:
    df = pd.read_csv(INPUT_PATH, header=None, na_values=["?"])
    if df.shape[1] <= max([AGE_INDEX] + ECG_ARR_COLS):
        raise ValueError("Too few columns with comma separator")
except Exception:
    df = pd.read_csv(INPUT_PATH, header=None, sep=";", na_values=["?"])

# --- compute features ---
age = pd.to_numeric(df.iloc[:, AGE_INDEX], errors="coerce")

ecg = df[ECG_ARR_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
ecg = ecg.clip(lower=0, upper=1)

df["ECG_ARR_COUNT"] = ecg.sum(axis=1)
df["ECG_ARR_ANY"] = (df["ECG_ARR_COUNT"] > 0).astype(int)

df["AGE_X_ECG_ARR_COUNT"] = age * df["ECG_ARR_COUNT"]
df["ELDERLY_X_ECG_ARR_ANY"] = ((age >= 65) & (df["ECG_ARR_ANY"] == 1)).astype(int)

df.to_csv(OUTPUT_PATH, index=False)
print("Wrote:", OUTPUT_PATH)
print("Added:", ["ECG_ARR_COUNT", "ECG_ARR_ANY", "AGE_X_ECG_ARR_COUNT", "ELDERLY_X_ECG_ARR_ANY"])