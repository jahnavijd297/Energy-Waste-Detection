
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

RESULTS_FILE = OUTPUTS / "classifier_anomaly_results.csv"
METRICS_FILE = OUTPUTS / "model_metrics.csv"

st.set_page_config(
    page_title="Energy Waste Detection",
    page_icon="⚡",
    layout="wide",
)

# --------------------------------------------------
# 2. DASHBOARD HEADER
# --------------------------------------------------

st.title("⚡ Energy Consumption Prediction")
st.subheader("Intelligent Energy-Waste Detection System")
st.caption(
    "Machine Learning | Energy Forecasting | Anomaly Detection"
)

st.divider()

# --------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# --------------------------------------------------

if not RESULTS_FILE.exists():
    st.error(
        "Anomaly results file was not found. "
        "Please run src/anomaly_model.py first."
    )
    st.stop()

try:
    df = pd.read_csv(RESULTS_FILE)
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read the results file: {error}")
    st.stop()

required_columns = [
    "building_id",
    "timestamp",
    "meter_reading",
    "anomaly",
    "predicted_anomaly",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing columns in results file: {missing_columns}")
    st.stop()

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

for column in [
    "meter_reading",
    "anomaly",
    "predicted_anomaly",
]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=required_columns).copy()

df["anomaly"] = df["anomaly"].astype(int)
df["predicted_anomaly"] = df["predicted_anomaly"].astype(int)

df["status"] = df["predicted_anomaly"].map({
    0: "Normal",
    1: "Possible anomaly",
})

if df.empty:
    st.error("The results file contains no valid readings.")
    st.stop()

# --------------------------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("Dashboard Controls")
st.sidebar.write("Filter the historical dataset below.")

all_buildings = sorted(df["building_id"].unique().tolist())

selected_buildings = st.sidebar.multiselect(
    "Select building IDs",
    options=all_buildings,
    default=all_buildings[:min(10, len(all_buildings))],
)

if not selected_buildings:
    st.warning("Select at least one building from the sidebar.")
    st.stop()

filtered = df[
    df["building_id"].isin(selected_buildings)
].copy()

if filtered.empty:
    st.warning("No readings found for the selected buildings.")
    st.stop()

# --------------------------------------------------
# 5. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

total_readings = len(filtered)

flagged_count = int(
    (filtered["predicted_anomaly"] == 1).sum()
)

actual_anomalies = int(
    (filtered["anomaly"] == 1).sum()
)

predicted_normal = total_readings - flagged_count

st.markdown("### Energy Monitoring Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Readings Analyzed",
    f"{total_readings:,}",
)

col2.metric(
    "Possible Anomalies",
    f"{flagged_count:,}",
)

col3.metric(
    "Actual Labelled Anomalies",
    f"{actual_anomalies:,}",
)

col4.metric(
    "Predicted Normal",
    f"{predicted_normal:,}",
)

st.info(
    "This dashboard currently displays historical building-level "
    "dataset results. It does not represent live electricity use "
    "from an individual college building."
)

st.divider()

# --------------------------------------------------
# 6. HISTORICAL ENERGY CONSUMPTION
# --------------------------------------------------

st.markdown("### 1. Historical Energy Consumption")

chart_data = filtered.sort_values("timestamp")

consumption_fig = px.line(
    chart_data,
    x="timestamp",
    y="meter_reading",
    color="building_id",
    title="Energy Meter Readings Over Time",
    labels={
        "timestamp": "Date and Time",
        "meter_reading": "Meter Reading",
        "building_id": "Building ID",
    },
)

consumption_fig.update_layout(
    hovermode="x unified",
    legend_title_text="Building",
)

st.plotly_chart(
    consumption_fig,
    use_container_width=True,
)

# --------------------------------------------------
# 7. ANOMALY DETECTION
# --------------------------------------------------

st.markdown("### 2. Intelligent Anomaly Detection")

anomalies = filtered[
    filtered["predicted_anomaly"] == 1
].copy()

if anomalies.empty:
    st.success(
        "No possible anomalies were flagged for this selection."
    )
else:
    anomaly_fig = px.scatter(
        anomalies,
        x="timestamp",
        y="meter_reading",
        color="status",
        hover_data=[
            "building_id",
            "anomaly",
        ],
        title="Readings Flagged by the ML Classifier",
        labels={
            "timestamp": "Date and Time",
            "meter_reading": "Meter Reading",
            "building_id": "Building ID",
            "anomaly": "Actual Dataset Label",
            "status": "Model Prediction",
        },
    )

    st.plotly_chart(
        anomaly_fig,
        use_container_width=True,
    )

st.caption(
    "A possible anomaly is a reading flagged by the model. "
    "It is not, by itself, proof of energy waste."
)

st.divider()

# --------------------------------------------------
# 8. ACTUAL LABELS VS PREDICTIONS
# --------------------------------------------------

st.markdown("### 3. Actual Labels vs Model Predictions")

comparison = (
    filtered.groupby(
        ["anomaly", "predicted_anomaly"]
    )
    .size()
    .reset_index(name="count")
)

comparison["Actual Label"] = comparison["anomaly"].map({
    0: "Normal",
    1: "Anomaly",
})

comparison["Model Prediction"] = comparison[
    "predicted_anomaly"
].map({
    0: "Normal",
    1: "Possible anomaly",
})

comparison_fig = px.bar(
    comparison,
    x="Actual Label",
    y="count",
    color="Model Prediction",
    barmode="group",
    title="Classification Results",
)

st.plotly_chart(
    comparison_fig,
    use_container_width=True,
)

# --------------------------------------------------
# 9. MODEL EVALUATION
# --------------------------------------------------

st.markdown("### 4. Model Evaluation")

actual = filtered["anomaly"]
predicted = filtered["predicted_anomaly"]

precision = precision_score(
    actual,
    predicted,
    zero_division=0,
)

recall = recall_score(
    actual,
    predicted,
    zero_division=0,
)

f1 = f1_score(
    actual,
    predicted,
    zero_division=0,
)

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Anomaly Precision",
    f"{precision:.2%}",
)

metric2.metric(
    "Anomaly Recall",
    f"{recall:.2%}",
)

metric3.metric(
    "Anomaly F1-Score",
    f"{f1:.4f}",
)

with st.expander("View confusion matrix and classification report"):
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    st.write(
        "Confusion matrix: rows are actual labels; "
        "columns are model predictions."
    )

    st.dataframe(
        pd.DataFrame(
            matrix,
            index=["Actual Normal", "Actual Anomaly"],
            columns=["Predicted Normal", "Predicted Anomaly"],
        ),
        use_container_width=True,
    )

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
        output_dict=True,
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True,
    )

if METRICS_FILE.exists():
    st.markdown("#### Saved consumption-model metrics")

    try:
        saved_metrics = pd.read_csv(METRICS_FILE)
        st.dataframe(
            saved_metrics,
            use_container_width=True,
            hide_index=True,
        )
    except (OSError, pd.errors.ParserError) as error:
        st.warning(f"Could not load saved metrics: {error}")

st.divider()

# --------------------------------------------------
# 10. DETAILED PREDICTION RESULTS
# --------------------------------------------------

st.markdown("### 5. Detailed Prediction Results")

show_all = st.checkbox(
    "Show all readings instead of only flagged anomalies",
    value=False,
)

if show_all:
    table_data = filtered.copy()
else:
    table_data = anomalies.copy()

table_data = table_data.sort_values(
    "timestamp",
    ascending=False,
)

st.dataframe(
    table_data,
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------
# 11. DOWNLOAD RESULTS
# --------------------------------------------------

st.markdown("### 6. Export Results")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Results (CSV)",
    data=csv_data,
    file_name="energy_detection_results.csv",
    mime="text/csv",
)

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Project: ML-Based Energy Consumption Prediction and "
    "Intelligent Energy-Waste Detection System"
)

st.caption(
    "Data source: historical building-level energy dataset. "
    "Live regional electricity data has not yet been integrated."
)
