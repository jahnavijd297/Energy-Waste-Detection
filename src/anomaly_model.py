
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "energy_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

BASE_FEATURES = [
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

NEW_FEATURE = "same_as_previous"
TARGET = "anomaly"
RANDOM_STATE = 42


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def select_threshold(y_true, probabilities):
    """Choose the threshold using validation data only."""
    rows = []

    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        rows.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    best = max(
        rows,
        key=lambda row: (row["f1"], row["precision"]),
    )
    return best, rows


def evaluate_model(name, model, threshold, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    missed = int(((y_test == 1) & (predictions == 0)).sum())
    false_positives = int(((y_test == 0) & (predictions == 1)).sum())

    print(f"\n=== {name} ===")
    print(f"Threshold:  {threshold:.2f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
    )
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"F1-score:        {f1:.4f}")
    print(f"Missed anomalies: {missed}")
    print(f"False positives:  {false_positives}")

    return {
        "model": name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missed_anomalies": missed,
        "false_positives": false_positives,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    required = BASE_FEATURES + [TARGET, "timestamp", "meter_reading"]
    df = df.dropna(subset=required).copy()
    df[TARGET] = df[TARGET].astype(int)

    # Sort each building's readings before creating the feature.
    df = df.sort_values(
        ["building_id", "timestamp"]
    ).reset_index(drop=True)

    # Use the previous recorded reading from the SAME building.
    # This does not use the current or any future reading.
    previous_reading = df.groupby(
        "building_id"
    )["meter_reading"].shift(1)

    df[NEW_FEATURE] = (
        df["meter_reading"].eq(previous_reading)
    ).astype(int)

    # First row per building has no previous reading.
    # Exclude it rather than incorrectly marking it as a repeat.
    df = df[previous_reading.notna()].copy()

    # Chronological ordering for time-based splitting.
    df = df.sort_values(
        ["timestamp", "building_id"]
    ).reset_index(drop=True)

    # Split on unique timestamps so buildings at the same time
    # cannot leak across training, validation, and test sets.
    times = pd.Index(
        df["timestamp"].drop_duplicates().sort_values()
    )

    if len(times) < 10:
        raise ValueError("Not enough timestamps for three data splits.")

    train_end = int(len(times) * 0.70)
    validation_end = int(len(times) * 0.85)

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
                f"{name} split must contain both label classes."
            )

    overlap = (
        set(train_df["timestamp"]) & set(validation_df["timestamp"])
        | set(train_df["timestamp"]) & set(test_df["timestamp"])
        | set(validation_df["timestamp"]) & set(test_df["timestamp"])
    )

    print("EXPERIMENT: REPEATED-READING FEATURE")
    print("====================================")
    for name, part in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: {len(part):,} rows | "
            f"{part['timestamp'].min()} to "
            f"{part['timestamp'].max()} | "
            f"anomalies: {int(part[TARGET].sum()):,}"
        )

    print("Shared timestamps across splits:", len(overlap))
    print(
        "Repeated-reading feature prevalence by label (%):"
    )
    print(
        df.groupby(TARGET)[NEW_FEATURE]
        .mean()
        .mul(100)
        .round(2)
        .to_string()
    )

    baseline_features = BASE_FEATURES
    experimental_features = BASE_FEATURES + [NEW_FEATURE]

    y_train = train_df[TARGET]
    y_validation = validation_df[TARGET]
    y_test = test_df[TARGET]

    # Train both models on the same training period.
    baseline_model = make_model()
    experimental_model = make_model()

    print("\nTraining baseline model...")
    baseline_model.fit(
        train_df[baseline_features], y_train
    )

    print("Training model with repeated-reading feature...")
    experimental_model.fit(
        train_df[experimental_features], y_train
    )

    # Select each threshold independently using validation only.
    baseline_validation_prob = baseline_model.predict_proba(
        validation_df[baseline_features]
    )[:, 1]

    experimental_validation_prob = experimental_model.predict_proba(
        validation_df[experimental_features]
    )[:, 1]

    baseline_best, baseline_threshold_rows = select_threshold(
        y_validation, baseline_validation_prob
    )
    experimental_best, experimental_threshold_rows = select_threshold(
        y_validation, experimental_validation_prob
    )

    print("\nValidation-selected thresholds")
    print(
        f"Baseline: {baseline_best['threshold']:.2f} | "
        f"F1={baseline_best['f1']:.4f} | "
        f"Precision={baseline_best['precision']:.4f} | "
        f"Recall={baseline_best['recall']:.4f}"
    )
    print(
        f"Experimental: {experimental_best['threshold']:.2f} | "
        f"F1={experimental_best['f1']:.4f} | "
        f"Precision={experimental_best['precision']:.4f} | "
        f"Recall={experimental_best['recall']:.4f}"
    )

    # Final comparison on the same test rows.
    baseline_result = evaluate_model(
        "BASELINE MODEL",
        baseline_model,
        baseline_best["threshold"],
        test_df[baseline_features],
        y_test,
    )

    experimental_result = evaluate_model(
        "MODEL WITH REPEATED-READING FEATURE",
        experimental_model,
        experimental_best["threshold"],
        test_df[experimental_features],
        y_test,
    )

    comparison_rows = []
    for result in [baseline_result, experimental_result]:
        comparison_rows.append({
            "model": result["model"],
            "threshold": result["threshold"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "missed_anomalies": result["missed_anomalies"],
            "false_positives": result["false_positives"],
        })

    comparison = pd.DataFrame(comparison_rows)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save under experiment-specific names; preserve current artifacts.
    joblib.dump(
        experimental_model,
        MODEL_DIR / "anomaly_classifier_repeat_feature_experiment.joblib",
    )

    comparison.to_csv(
        OUTPUT_DIR / "repeat_feature_comparison.csv",
        index=False,
    )

    all_thresholds = pd.concat(
    [
        pd.DataFrame(baseline_threshold_rows).assign(
            model="baseline"
        ),
        pd.DataFrame(experimental_threshold_rows).assign(
            model="repeated_reading_feature"
        ),
    ],
    ignore_index=True,
)
    all_thresholds.to_csv(
        OUTPUT_DIR / "repeat_feature_validation_thresholds.csv",
        index=False,
    )

    prediction_output = test_df[
        ["building_id", "timestamp", "meter_reading", TARGET, NEW_FEATURE]
    ].copy()

    prediction_output["baseline_probability"] = (
        baseline_result["probabilities"]
    )
    prediction_output["baseline_prediction"] = (
        baseline_result["predictions"]
    )
    prediction_output["experimental_probability"] = (
        experimental_result["probabilities"]
    )
    prediction_output["experimental_prediction"] = (
        experimental_result["predictions"]
    )

    prediction_output.to_csv(
        OUTPUT_DIR / "repeat_feature_experiment_predictions.csv",
        index=False,
    )

    print("\n=== FINAL TEST COMPARISON ===")
    print(comparison.round(4).to_string(index=False))
    print("\nSaved experimental model and comparison files.")
    print("Existing model and dashboard prediction files were not overwritten.")


if __name__ == "__main__":
    main()
