# Models Documentation

This document describes the machine learning models used in the ATP forecaster project.

## XGBoost v1 (`xgb_v1`)

**Model Type:** XGBoost Classifier (binary classification)

**Saved Location:** `models/xgb_v1.pkl`

**Training Dataset:** `data/training_data/dataset_v1.parquet`

**Training Script:** `python -m atp_forecaster.scripts.tune_xgb_v1`

**Performance:** `Mean AUC=0.7315, Mean Accuracy=0.6663, Mean LogLoss=0.6044` with 20 fold CV

**Note:** The saved model is an untrained model.

## Transitive v1 (`xgb_v1`)

**Model Type:** Transitive, point by point

**Training Dataset:** `data/cleaned/atp_matches_cleaned.parquet`

**Performance:** `Mean AUC=0.47845, Mean Accuracy=0.4858, Mean LogLoss=0.71257` 2024 Validation Set

## Transitive v2 (`xgb_v1`)

**Model Type:** Transitive, point by point

**Training Dataset:** `data/cleaned/atp_matches_cleaned.parquet`

**Performance:** `Mean AUC=0.4775, Mean Accuracy=0.4725, Mean LogLoss=0.79869` 2024 Validation Set