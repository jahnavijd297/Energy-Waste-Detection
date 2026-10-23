
import pandas as pd


def select_model_features(df):
    """
    Select the input features used by the energy
    consumption prediction models.
    """

    feature_columns = [
        "building_id",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_24",
        "rolling_mean_24",
        "rolling_std_24",
    ]

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    X = df[feature_columns].copy()

    # Convert the building identifier to numeric form
    # when possible, so scikit-learn can use it.
    X["building_id"] = pd.to_numeric(
        X["building_id"], errors="coerce"
    )

    X = X.dropna()

    return X


def prepare_target(df):
    """
    Return actual energy readings for model training.
    """

    if "meter_reading" not in df.columns:
        raise ValueError(
            "The dataset must contain meter_reading."
        )

    y = pd.to_numeric(
        df["meter_reading"], errors="coerce"
    )

    return y
