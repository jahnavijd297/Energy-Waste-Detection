
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "outputs" / "predictions.csv"
OUTPUT_PATH = ROOT / "outputs" / "anomaly_results.csv"


def detect_anomalies():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Predictions file not found. Run training.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required = [
        "meter_reading",
        "predicted_reading",
        "anomaly",
    ]

    missing = [
        col for col in required if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Actual consumption minus predicted consumption.
    df["prediction_error"] = (
        df["meter_reading"] - df["predicted_reading"]
    )

    # Initial rule: flag readings more than 20% above
    # the non-negative predicted consumption.
    threshold = (
        df["predicted_reading"].clip(lower=0) * 0.20
    )

    df["predicted_anomaly"] = (
        (df["prediction_error"] > threshold)
        & (df["prediction_error"] > 0)
    ).astype(int)

    # Compare the rule with the dataset's existing labels.
    y_true = df["anomaly"].astype(int)
    y_pred = df["predicted_anomaly"]

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

    df["status"] = df["predicted_anomaly"].map({
        0: "Within threshold",
        1: "Possible high usage",
    })

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved anomaly results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    detect_anomalies()
