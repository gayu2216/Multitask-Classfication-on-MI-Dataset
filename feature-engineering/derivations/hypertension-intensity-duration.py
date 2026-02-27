import pandas as pd

INPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI.data"
OUTPUT_PATH = r"C:\Users\hung\Downloads\Multitask-Classfication-on-MI-Dataset\feature-engineering\MI_hemodynamic_instability.csv"

S_AD_KBRIG = 34
D_AD_KBRIG = 35
S_AD_ORIT  = 36
D_AD_ORIT  = 37

try:
    df = pd.read_csv(INPUT_PATH, header=None, na_values=["?"])
    if df.shape[1] <= max(S_AD_KBRIG, D_AD_KBRIG, S_AD_ORIT, D_AD_ORIT):
        raise ValueError("Too few columns with comma separator")
except Exception:
    df = pd.read_csv(INPUT_PATH, header=None, sep=";", na_values=["?"])

sbp_k = pd.to_numeric(df.iloc[:, S_AD_KBRIG], errors="coerce")
dbp_k = pd.to_numeric(df.iloc[:, D_AD_KBRIG], errors="coerce")
sbp_i = pd.to_numeric(df.iloc[:, S_AD_ORIT],  errors="coerce")
dbp_i = pd.to_numeric(df.iloc[:, D_AD_ORIT],  errors="coerce")

df["HYPOTENSION_KBRIG"] = (sbp_k < 90).astype(int)
df["HYPERTENSION_CRISIS_KBRIG"] = (sbp_k > 180).astype(int)

df["PULSE_PRESSURE_KBRIG"] = sbp_k - dbp_k
df["PULSE_PRESSURE_ORIT"] = sbp_i - dbp_i

df["SBP_DELTA_TO_ICU"] = sbp_i - sbp_k
df["DBP_DELTA_TO_ICU"] = dbp_i - dbp_k

df.to_csv(OUTPUT_PATH, index=False)
print("Wrote:", OUTPUT_PATH)
print("Added:", [
    "HYPOTENSION_KBRIG", "HYPERTENSION_CRISIS_KBRIG",
    "PULSE_PRESSURE_KBRIG", "PULSE_PRESSURE_ORIT",
    "SBP_DELTA_TO_ICU", "DBP_DELTA_TO_ICU"
])