import pandas as pd

INPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"
OUTPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI_missing_k_blood.csv"

LAB_INDEX = 83  # attribute #84 (K_BLOOD)

try:
    df = pd.read_csv(INPUT_PATH, header=None)
    if df.shape[1] <= LAB_INDEX:
        raise ValueError("Too few columns with comma separator")
except Exception:
    df = pd.read_csv(INPUT_PATH, header=None, sep=";")

if df.shape[1] <= LAB_INDEX:
    raise IndexError(f"Loaded shape {df.shape}. LAB_INDEX={LAB_INDEX} out of bounds. Wrong delimiter?")

s = df.iloc[:, LAB_INDEX]

# missing if NaN OR literal '?' OR empty string
missing = s.isna() | (s.astype(str).str.strip().isin(["", "?"]))
df[df.shape[1]] = missing.astype("Int64")  # append

df.to_csv(OUTPUT_PATH, index=False, header=False)
print(f"OK: wrote {OUTPUT_PATH} (appended LAB_MISSING_K_BLOOD)")