from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "energy_data.csv"
OUT = ROOT / "outputs" / "flatline_24h_verification"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["building_id", "timestamp", "meter_reading"])
df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

# A flatline must span at least 24 elapsed hours.
predictions = []
for _, group in df.groupby("building_id", sort=False):
    last_time = None
    last_reading = None
    run_length = 0
    flags = []

    for row in group.itertuples(index=False):
        ts = row.timestamp
        reading = row.meter_reading

        if last_time is not None:
            elapsed = (ts - last_time).total_seconds() / 3600
        else:
            elapsed = None

        if (
            last_time is not None
            and elapsed == 1
            and reading == last_reading
        ):
            run_length += 1
        else:
            run_length = 1

        flags.append(int(run_length >= 25))

        last_time = ts
        last_reading = reading

    predictions.extend(flags)

df["flatline_24_observations"] = predictions
df["reading_equals_1"] = df["meter_reading"].eq(1.0).astype(int)
df["hybrid_prediction"] = (
    (df["reading_equals_1"] == 1)
    | (df["flatline_24_observations"] == 1)
).astype(int)

result_path = OUT / "predictions.csv"
df.to_csv(result_path, index=False)

print("Rows evaluated:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print("24-observation flatline flags:", int(df["flatline_24_observations"].sum()))
print("Hybrid flags:", int(df["hybrid_prediction"].sum()))
print("Saved:", result_path)

if "anomaly" in df.columns:
    valid = df["anomaly"].notna()
    y = df.loc[valid, "anomaly"].astype(int)
    p = df.loc[valid, "hybrid_prediction"]
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    print("\nFULL-DATA DIAGNOSTIC ONLY — NOT UNBIASED TEST RESULTS")
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print("Precision:", round(precision_score(y, p, zero_division=0), 4))
    print("Recall:", round(recall_score(y, p, zero_division=0), 4))
    print("F1:", round(f1_score(y, p, zero_division=0), 4))
