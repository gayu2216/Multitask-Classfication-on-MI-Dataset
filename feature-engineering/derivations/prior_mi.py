import pandas as pd

INPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"
OUTPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI_prior_mi_flag.csv"

INF_ANAM_INDEX = 3  # attribute #4

try:
    df = pd.read_csv(INPUT_PATH, header=None)
    if df.shape[1] <= INF_ANAM_INDEX:
        raise ValueError("Too few columns with comma separator")
except Exception:
    df = pd.read_csv(INPUT_PATH, header=None, sep=";")

if df.shape[1] <= INF_ANAM_INDEX:
    raise IndexError(f"Loaded shape {df.shape}. INF_ANAM_INDEX={INF_ANAM_INDEX} out of bounds. Wrong delimiter?")

inf = pd.to_numeric(df.iloc[:, INF_ANAM_INDEX], errors="coerce")
df[df.shape[1]] = (inf > 0).astype("Int64")  # append

df.to_csv(OUTPUT_PATH, index=False, header=False)
print(f"OK: wrote {OUTPUT_PATH} (appended PRIOR_MI)")