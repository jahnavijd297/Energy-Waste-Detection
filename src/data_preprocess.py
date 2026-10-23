
import numpy as np
import pandas as pd


def preprocess(df):
    """
    Clean energy readings and create features
    for energy consumption prediction.
    """

    df = df.copy()

    # Check required columns
    required_columns = [
        "building_id",
        "timestamp",
        "meter_reading",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Convert data to correct types
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df["meter_reading"] = pd.to_numeric(
        df["meter_reading"], errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=required_columns
    )

    # Sort readings by building and time
    df = df.sort_values(
        ["building_id", "timestamp"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid readings found.")

    # Extract time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    # Previous consumption readings
    grouped = df.groupby("building_id")[
        "meter_reading"
    ]

    df["lag_1"] = grouped.shift(1)
    df["lag_2"] = grouped.shift(2)
    df["lag_3"] = grouped.shift(3)
    df["lag_24"] = grouped.shift(24)

    # Previous 24-reading average and variation
    df["rolling_mean_24"] = grouped.transform(
        lambda x: x.shift(1).rolling(24).mean()
    )

    df["rolling_std_24"] = grouped.transform(
        lambda x: x.shift(1).rolling(24).std()
    )

    # Encode repeating time patterns
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

    # Remove rows without sufficient history
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("Energy data preprocessing module is ready.")
    print("Required columns:")
    print("building_id, timestamp, meter_reading")
