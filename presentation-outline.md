# Presentation Outline — Failure Before Failure

## Slide 1 — Title

**Failure Before Failure**  
AI-powered industrial predictive maintenance learning prototype.

Speaker note: This is an educational prototype using synthetic research data, not a certified safety system.

## Slide 2 — Problem and goal

- Unexpected industrial downtime is costly.
- Explore sensor-based indicators that can help people investigate earlier.
- Goal: show evidence and checks, not guarantee a prediction.

## Slide 3 — Scope and data

- Industrial machinery only.
- UCI AI4I 2020 dataset: 10,000 synthetic records.
- Imbalanced target: 339 machine-failure rows and 9,661 non-failure rows.
- Cite UCI and the CC BY 4.0 license.

## Slide 4 — System architecture

- Pandas/NumPy data preparation.
- Scikit-learn failure classifier.
- PyTorch Autoencoder for unusual patterns.
- Transparent Python investigator.
- FastAPI, SQLite, and React dashboard.
- No hosted AI API key.

## Slide 5 — Data preparation

- Selected six sensor/product inputs.
- Excluded identifiers and failure-mode labels to reduce leakage.
- Stratified train/test split.
- Fit scaling and encoding within the training pipeline.

## Slide 6 — Model results

- Random Forest: 0.982 accuracy, 0.799 balanced accuracy, 0.603 failure recall.
- Random Forest confusion matrix: `[[1923, 9], [27, 41]]`.
- Autoencoder: 6.21% held-out normal false-alarm rate and 0.235 failure recall.
- Highlight why accuracy alone can mislead.

## Slide 7 — Investigator and dashboard

- Shows sensor values, model outputs, threshold comparisons, rule evidence, and suggested checks.
- Dashboard supports human maintenance-action records.
- Dataset rules are not universal factory limits.

## Slide 8 — Limitations and conclusion

- Synthetic data; not validated on live equipment.
- Anomaly is not proof of failure.
- Models miss some failure-labelled rows and can produce false alarms.
- Human review is essential; safety-critical use is out of scope.

## Slide 9 — Demonstration

- Open the dashboard and select a machine.
- Submit a sample reading and inspect analysis/evidence.
- Record a human-reviewed maintenance action.
- See [the demo script](demo-script.md) for commands and sample data.
