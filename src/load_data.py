
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "test_df.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_dataset():
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA}"
        )

    df = pd.read_csv(RAW_DATA)

    # Remove the saved DataFrame index, if present.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required = [
        "building_id",
        "timestamp",
        "meter_reading",
        "anomaly",
    ]

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Convert columns to their expected types.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df["meter_reading"] = pd.to_numeric(
        df["meter_reading"], errors="coerce"
    )
    df["anomaly"] = pd.to_numeric(
        df["anomaly"], errors="coerce"
    )

    df = df.dropna(subset=required)

    df = df.sort_values(
        ["building_id", "timestamp"]
    ).reset_index(drop=True)

    # Create time features.
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Create lag features separately for each building.
    grouped = df.groupby("building_id")["meter_reading"]

    df["lag_1"] = grouped.shift(1)
    df["lag_2"] = grouped.shift(2)
    df["lag_3"] = grouped.shift(3)
    df["lag_24"] = grouped.shift(24)

    df["rolling_mean_24"] = grouped.transform(
        lambda x: x.shift(1).rolling(24).mean()
    )

    df["rolling_std_24"] = grouped.transform(
        lambda x: x.shift(1).rolling(24).std()
    )

    # Encode recurring time patterns.
    import numpy as np

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )
    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )
    df["month_sin"] = np.sin(
        2 * np.pi * (df["month"] - 1) / 12
    )
    df["month_cos"] = np.cos(
        2 * np.pi * (df["month"] - 1) / 12
    )

    # Keep anomaly labels; drop rows lacking history.
    df = df.dropna().reset_index(drop=True)
    df["anomaly"] = df["anomaly"].astype(int)

    PROCESSED_DIR.mkdir(
        parents=True, exist_ok=True
    )

    output_path = PROCESSED_DIR / "energy_data.csv"
    df.to_csv(output_path, index=False)

    print("Dataset preprocessing completed!")
    print("Rows and columns:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Anomaly label counts:")
    print(df["anomaly"].value_counts().to_string())
    print(f"Saved dataset to: {output_path}")


if __name__ == "__main__":
    load_dataset()
