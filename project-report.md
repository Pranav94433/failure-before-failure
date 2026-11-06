# Failure Before Failure — Project Report

## Abstract

Failure Before Failure is a local-first learning prototype for industrial predictive maintenance. It validates sensor readings, estimates a known failure label with a supervised classifier, flags unusual input patterns with a PyTorch Autoencoder, and presents transparent evidence and maintenance checks through FastAPI and a React dashboard. SQLite stores readings, analyses, and human-reviewed actions. No hosted language-model API is required.

## Problem and scope

Unexpected industrial equipment downtime can disrupt production. This prototype explores how sensor readings might support earlier human investigation. It is limited to industrial machinery and is not a certified industrial safety system. It does not claim that a detected anomaly proves failure or that a model guarantees advance warning.

## Architecture

```text
Sensor input / AI4I dataset
        |
        +--> Pandas validation and feature selection
        |       |
        |       +--> Scikit-learn failure classifier
        |       +--> PyTorch normal-pattern Autoencoder
        |
        +--> Transparent AI4I evidence rules
                    |
                    v
             FastAPI analysis API
                    |
             SQLite local storage
                    |
             React dashboard
                    |
             Human maintenance review
```

The classifier and Autoencoder use separate saved artifacts. The Autoencoder threshold is selected from normal validation examples. The investigator reports observed inputs, dataset-rule matches, and model outputs without asserting that a rule caused a real-world fault.

## Dataset

The project uses the UCI AI4I 2020 Predictive Maintenance Dataset: 10,000 synthetic rows with product type, air/process temperature, rotational speed, torque, tool wear, a machine-failure label, and five failure-mode labels. The dataset reports no missing values. The machine-failure target has 339 positive and 9,661 negative rows, so the classes are imbalanced.

The project keeps `UDI`, `Product ID`, and failure-mode labels out of the selected classifier inputs. The selected inputs are product `Type`, air temperature, process temperature, rotational speed, torque, and tool wear. Data is split before preprocessing; the scaler and encoder are fit only on training data.

**Citation:** AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. <https://doi.org/10.24432/C5HS5C>. Licensed CC BY 4.0.

## Models and evaluation

The classifier workflow compares a majority-class baseline and a balanced logistic-regression baseline, then trains a class-balanced Random Forest. A stratified 80/20 split holds out 2,000 records. The Autoencoder trains on normal examples from the development split, uses normal validation errors to select its threshold, and evaluates on the separate held-out set.

Results from the recorded workspace run:

| Metric | Random Forest | Autoencoder |
|---|---:|---:|
| Accuracy | 0.982 | — |
| Balanced accuracy | 0.799 | 0.587 |
| Positive precision | 0.820 | 0.118 |
| Failure recall | 0.603 | 0.235 |
| F1 | 0.695 | — |
| Average precision | 0.762 | 0.188 |
| ROC AUC | 0.965 | 0.757 |

The Random Forest confusion matrix was `[[1923, 9], [27, 41]]`, meaning 27 actual failure-labelled test rows were missed. The Autoencoder used a 0.165453 reconstruction-error threshold (the 95th percentile of normal validation errors); its held-out normal false-alarm rate was 6.21% (120/1932), and it detected 16 of 68 failure-labelled test rows.

These are single-run results on synthetic data. They are not independent validation, real-world performance claims, or proof of operational suitability. The high Random Forest accuracy should be considered alongside its missed failures and class imbalance.

## Investigator and API

The investigator checks documented AI4I synthetic patterns:

- HDF pattern: process-air temperature difference under 8.6 K with speed below 1,380 rpm.
- PWF pattern: torque and speed imply power below 3,500 W or above 9,000 W.
- OSF pattern: tool-wear/torque product exceeds the product-type-specific dataset threshold.
- TWF wear band: tool wear is within the dataset's documented 200–240 minute range.

The service also reports when the Autoencoder reconstruction error exceeds its validation threshold. A classifier estimate is shown as a model output and is not treated as a calibrated real-world failure probability.

FastAPI validates request fields, serves analysis and machine-history endpoints, and records data in SQLite. The dashboard presents those results, trends, suggestions, and human-entered maintenance actions. Analysis API integration has been manually verified against the local server.

## Limitations and responsible use

1. AI4I is synthetic and may not represent the machines, sensors, failure rates, or operating conditions of a real factory.
2. The data rows do not establish a validated live-machine time-series monitoring setup.
3. The held-out sample is small for failure cases, and model outputs are not calibrated.
4. A false alarm can create unnecessary work; a missed failure can create false reassurance.
5. The investigator's thresholds mirror dataset construction rules and must not be used as universal industrial thresholds.
6. A qualified human must compare evidence with machine-specific documentation and decide any maintenance action.
7. No live machine control, safety interlock, or autonomous maintenance action is implemented.

## Future work

Before any real-world use, obtain appropriate machine-specific data, document sensor units and provenance, evaluate temporal and equipment-level splits, calibrate outputs, measure false alarms and missed detections with domain experts, and complete security, reliability, and safety reviews. The planned roadmap also includes richer history/trend analysis and presentation materials; the current dashboard is a local prototype.
