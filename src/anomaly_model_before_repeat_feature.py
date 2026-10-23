
from pathlib import Path

import joblib
import numpy as np
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


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    df = df.dropna(
        subset=FEATURES + [TARGET, "timestamp"]
    ).copy()

    df[TARGET] = df[TARGET].astype(int)
    df = df.sort_values(
        ["timestamp", "building_id"]
    ).reset_index(drop=True)

    # Split using unique timestamps to prevent boundary overlap.
    times = pd.Index(
        df["timestamp"].drop_duplicates().sort_values()
    )

    if len(times) < 10:
        raise ValueError("Not enough timestamps for three data splits.")

    train_end = int(len(times) * 0.70)
    validation_end = int(len(times) * 0.85)

    train_end = max(1, min(train_end, len(times) - 2))
    validation_end = max(
        train_end + 1,
        min(validation_end, len(times) - 1),
    )

    train_times = set(times[:train_end])
    validation_times = set(times[train_end:validation_end])
    test_times = set(times[validation_end:])

    train_df = df[df["timestamp"].isin(train_times)].copy()
    validation_df = df[
        df["timestamp"].isin(validation_times)
    ].copy()
    test_df = df[df["timestamp"].isin(test_times)].copy()

    for name, part in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        if part.empty or part[TARGET].nunique() < 2:
            raise ValueError(
                f"{name} data must contain both normal and anomaly "
                "labels. Check the date ranges and dataset."
            )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURES]
    y_validation = validation_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("Chronological data split")
    print(
        f"Training:   {len(train_df):,} rows, "
        f"{train_df['timestamp'].min()} to "
        f"{train_df['timestamp'].max()}"
    )
    print(
        f"Validation: {len(validation_df):,} rows, "
        f"{validation_df['timestamp'].min()} to "
        f"{validation_df['timestamp'].max()}"
    )
    print(
        f"Test:       {len(test_df):,} rows, "
        f"{test_df['timestamp'].min()} to "
        f"{test_df['timestamp'].max()}"
    )

    overlap = (
        set(train_df["timestamp"]) & set(validation_df["timestamp"])
        | set(train_df["timestamp"]) & set(test_df["timestamp"])
        | set(validation_df["timestamp"]) & set(test_df["timestamp"])
    )
    print("Shared timestamps across splits:", len(overlap))

    # Train only on the training period.
    model = make_model()

    print("\nTraining Random Forest classifier...")
    model.fit(X_train, y_train)

    # Select the threshold on validation data only.
    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    candidate_thresholds = np.arange(0.05, 0.96, 0.01)
    threshold_results = []

    for threshold in candidate_thresholds:
        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )

        threshold_results.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # Choose the threshold with the best validation F1-score.
    # If scores tie, prefer the higher precision.
    best = max(
        threshold_results,
        key=lambda row: (row["f1"], row["precision"]),
    )
    best_threshold = best["threshold"]

    print("\nValidation-selected threshold:", round(best_threshold, 2))
    print(f"Validation precision: {best['precision']:.4f}")
    print(f"Validation recall:    {best['recall']:.4f}")
    print(f"Validation F1-score:  {best['f1']:.4f}")

    # Evaluate both thresholds on the same untouched test set.
    test_probabilities = model.predict_proba(X_test)[:, 1]

    default_predictions = (
        test_probabilities >= 0.50
    ).astype(int)
    tuned_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print("\n=== Test results: default threshold 0.50 ===")
    print(
        confusion_matrix(
            y_test, default_predictions, labels=[0, 1]
        )
    )
    print(
        classification_report(
            y_test,
            default_predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    print("\n=== Test results: validation-tuned threshold ===")
    print(
        confusion_matrix(
            y_test, tuned_predictions, labels=[0, 1]
        )
    )
    print(
        classification_report(
            y_test,
            tuned_predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )

    print("Default test precision:",
          round(precision_score(y_test, default_predictions, zero_division=0), 4))
    print("Default test recall:",
          round(recall_score(y_test, default_predictions, zero_division=0), 4))
    print("Default test F1:",
          round(f1_score(y_test, default_predictions, zero_division=0), 4))

    print("Tuned test precision:",
          round(precision_score(y_test, tuned_predictions, zero_division=0), 4))
    print("Tuned test recall:",
          round(recall_score(y_test, tuned_predictions, zero_division=0), 4))
    print("Tuned test F1:",
          round(f1_score(y_test, tuned_predictions, zero_division=0), 4))

    default_missed = int(
        ((y_test == 1) & (default_predictions == 0)).sum()
    )
    tuned_missed = int(
        ((y_test == 1) & (tuned_predictions == 0)).sum()
    )

    print("\nActual test anomalies:", int(y_test.sum()))
    print("Missed with default threshold:", default_missed)
    print("Missed with tuned threshold:", tuned_missed)

    # Save the model and predictions made on the untouched test period.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "anomaly_classifier.joblib"
    joblib.dump(model, model_path)

    results = test_df[
        ["building_id", "timestamp", "meter_reading", TARGET]
    ].copy()

    results["anomaly_probability"] = test_probabilities
    results["predicted_anomaly_default"] = default_predictions
    results["predicted_anomaly"] = tuned_predictions
    results["status"] = pd.Series(
        tuned_predictions, index=results.index
    ).map({
        0: "Normal",
        1: "Possible anomaly",
    })

    results_path = OUTPUT_DIR / "classifier_anomaly_results.csv"
    results.to_csv(results_path, index=False)

    threshold_path = OUTPUT_DIR / "anomaly_threshold_results.csv"
    pd.DataFrame(threshold_results).to_csv(
        threshold_path, index=False
    )

    print("\nSaved model:", model_path)
    print("Saved test predictions:", results_path)
    print("Saved validation threshold results:", threshold_path)


if __name__ == "__main__":
    main()
