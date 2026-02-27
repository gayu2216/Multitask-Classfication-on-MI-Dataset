import pandas as pd

INPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"
OUTPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI_elderly_flag.csv"

AGE_INDEX = 1  # attribute #2

# Try comma first; fall back to semicolon if needed
try:
    df = pd.read_csv(INPUT_PATH, header=None)
    if df.shape[1] <= AGE_INDEX:
        raise ValueError("Too few columns with comma separator")
except Exception:
    df = pd.read_csv(INPUT_PATH, header=None, sep=";")

if df.shape[1] <= AGE_INDEX:
    raise IndexError(f"Loaded shape {df.shape}. AGE_INDEX={AGE_INDEX} out of bounds. Wrong delimiter?")

age = pd.to_numeric(df.iloc[:, AGE_INDEX], errors="coerce")
df[df.shape[1]] = (age >= 65).astype("Int64")  # append

df.to_csv(OUTPUT_PATH, index=False, header=False)
print(f"OK: wrote {OUTPUT_PATH} (appended ELDERLY_65)")