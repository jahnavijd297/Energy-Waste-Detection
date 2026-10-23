
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
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

TARGET = "anomaly"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET, "timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Chronological split: older data trains; newer data tests.
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one anomaly class. "
            "A classifier cannot learn both classes from this split."
        )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training Random Forest anomaly classifier...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "anomaly_classifier.joblib"
    joblib.dump(model, model_path)

    results = df.iloc[split_index:][
        ["building_id", "timestamp", "meter_reading", TARGET]
    ].copy()
    results["predicted_anomaly"] = predictions
    results["status"] = results["predicted_anomaly"].map({
        0: "Normal",
        1: "Possible anomaly",
    })

    results_path = OUTPUT_DIR / "classifier_anomaly_results.csv"
    results.to_csv(results_path, index=False)

    print("\nModel saved to:", model_path)
    print("Predictions saved to:", results_path)


if __name__ == "__main__":
    main()
