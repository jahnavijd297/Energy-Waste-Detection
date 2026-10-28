
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "energy_data.csv"
CLASSIFIER_FILE = ROOT / "outputs" / "classifier_anomaly_results.csv"
OUTPUT_FILE = ROOT / "outputs" / "hybrid_rule_candidate.csv"


def calculate_flatlines(data: pd.DataFrame) -> pd.DataFrame:
    """Flag runs of 25+ identical hourly readings, resetting at time gaps."""
    data = data.sort_values(
        ["building_id", "timestamp"]
    ).copy()

    flags = pd.Series(0, index=data.index, dtype="int8")

    for _, group in data.groupby("building_id", sort=False):
        run_length = 0
        previous_time = None
        previous_reading = None

        for index, row in group.iterrows():
            timestamp = row["timestamp"]
            reading = row["meter_reading"]

            continuous = (
                previous_time is not None
                and (timestamp - previous_time).total_seconds() == 3600
            )
            same_reading = (
                previous_time is not None
                and reading == previous_reading
            )

            if continuous and same_reading:
                run_length += 1
            else:
                run_length = 1

            flags.at[index] = int(run_length >= 25)
            previous_time = timestamp
            previous_reading = reading

    data["flatline_24h"] = flags
    return data


def main() -> None:
    data = pd.read_csv(DATA_FILE)
    data["timestamp"] = pd.to_datetime(
        data["timestamp"], errors="coerce"
    )
    data = data.dropna(
        subset=["building_id", "timestamp", "meter_reading"]
    )

    data = calculate_flatlines(data)

    # Use the existing classifier output as the dashboard row base.
    result = pd.read_csv(CLASSIFIER_FILE)
    result["timestamp"] = pd.to_datetime(
        result["timestamp"], errors="coerce"
    )

    # Merge only the flatline flag to avoid duplicating meter_reading.
    rule_columns = data[
        ["building_id", "timestamp", "flatline_24h"]
    ]

    result = result.merge(
        rule_columns,
        on=["building_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )

    if result["flatline_24h"].isna().any():
        missing = int(result["flatline_24h"].isna().sum())
        raise ValueError(
            f"{missing} classifier rows did not match the source dataset."
        )

    result["flatline_24h"] = result["flatline_24h"].astype(int)

    # Hybrid rule: exact reading of 1 OR a 24-hour flatline.
    result["hybrid_prediction"] = (
        result["meter_reading"].eq(1.0)
        | result["flatline_24h"].eq(1)
    ).astype(int)

    # Keep dashboard-facing predictions consistent with the hybrid rule.
    result["predicted_anomaly"] = result["hybrid_prediction"]
    result["status"] = result["predicted_anomaly"].map(
        {0: "Normal", 1: "Possible anomaly"}
    )
    result["hybrid_status"] = result["hybrid_prediction"].map(
        {0: "Normal", 1: "Possible anomaly"}
    )

    columns = [
        "building_id",
        "timestamp",
        "meter_reading",
        "anomaly",
        "anomaly_probability",
        "predicted_anomaly_default",
        "predicted_anomaly",
        "status",
        "flatline_24h",
        "hybrid_prediction",
        "hybrid_status",
    ]

    result = result[columns].sort_values(
        ["timestamp", "building_id"]
    )
    result.to_csv(OUTPUT_FILE, index=False)

    print("Rows generated:", len(result))
    print(
        "Date range:",
        result["timestamp"].min(),
        "to",
        result["timestamp"].max(),
    )
    print(
        "Reading equals 1:",
        int(result["meter_reading"].eq(1.0).sum()),
    )
    print("Flatline flags:", int(result["flatline_24h"].sum()))
    print("Hybrid flags:", int(result["hybrid_prediction"].sum()))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
