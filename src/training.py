
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "energy_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

FEATURES = [
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

TARGET = "meter_reading"


def train_models():
    df = pd.read_csv(DATA_PATH)

    df = df.sort_values(
        ["timestamp", "building_id"]
    ).reset_index(drop=True)

    df = df.dropna(subset=FEATURES + [TARGET])

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    predictions_df = test_df[
        ["building_id", "timestamp", TARGET, "anomaly"]
    ].copy()

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results.append({
            "model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": mean_squared_error(
                y_test, predictions
            ) ** 0.5,
            "R2": r2_score(y_test, predictions),
        })

        joblib.dump(
            model,
            MODEL_DIR / f"{name}.joblib",
        )

        if name == "random_forest":
            predictions_df["predicted_reading"] = predictions

    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    print("\nModel evaluation:")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nSaved models to:", MODEL_DIR)
    print("Saved predictions to:", OUTPUT_DIR / "predictions.csv")


if __name__ == "__main__":
    train_models()
