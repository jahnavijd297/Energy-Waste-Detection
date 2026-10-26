# Energy Waste Detection — Experiment Log

## Current decision
Keep the existing model and dashboard unchanged.
The 24-hour hybrid flatline rule is an experimental candidate only.
Do not integrate it until real-time behavior and fresh-period performance are verified.

## Experiments and findings

### 1. Baseline rule
Rule: flag when meter_reading == 1.0

Validation:
- TP: 91
- FP: 0
- FN: 154
- Precision: 1.000
- Recall: 0.371
- F1: 0.542

### 2. Repeated reading of 61.0
A repeated-61 rule showed a strong association with anomalies in building 1143 in November 2016.
The association was localized and should not be treated as a general rule.

### 3. Flatline experiments
Consecutive identical hourly readings were associated with a higher anomaly rate as run length increased.
However, assigning the final run length to earlier rows can leak future information.
Use only the run duration known at the current prediction time.

### 4. Six-hour hybrid
Rule: meter_reading == 1.0 OR unchanged for at least 6 consecutive hourly observations.

Validation:
- TP: 160
- FP: 202
- FN: 85
- Precision: 0.442
- Recall: 0.653
- F1: 0.527

Previously inspected test:
- TP: 119
- FP: 15
- FN: 7
- Precision: 0.888
- Recall: 0.944
- F1: 0.915

The test set was repeatedly inspected during experimentation. Do not report this as unbiased final performance.

### 5. Threshold comparison on validation
| Rule | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid >= 24 hours | 117 | 0 | 128 | 1.000 | 0.478 | 0.646 |
| Hybrid >= 18 hours | 126 | 54 | 119 | 0.700 | 0.514 | 0.593 |
| Hybrid >= 12 hours | 138 | 117 | 107 | 0.541 | 0.563 | 0.552 |
| Baseline reading == 1.0 | 91 | 0 | 154 | 1.000 | 0.371 | 0.542 |
| Hybrid >= 6 hours | 160 | 202 | 85 | 0.442 | 0.653 | 0.527 |
| Hybrid >= 8 hours | 146 | 165 | 99 | 0.469 | 0.596 | 0.525 |

## Next steps
1. Inspect which model and prediction logic the dashboard currently loads.
2. Verify the flatline rule handles missing timestamps and uses only past/current observations.
3. Evaluate the selected candidate on a fresh, untouched chronological period.
4. Compare precision, recall, F1, and false positives before integrating anything.
5. Only after verification, decide whether to update the dashboard.
