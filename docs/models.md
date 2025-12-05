# Models Documentation

This document describes the machine learning models used in the ATP forecaster project.

## XGBoost v1 (`xgb_v1`)

**Model Type:** XGBoost Classifier (binary classification)

**Saved Location:** `models/xgb_v1.pkl`

**Training Dataset:** `data/training_data/dataset_v1.parquet`

**Training Script:** `python -m atp_forecaster.scripts.tune_xgb_v1`

**Performance:** `log_loss: 0.5233977167235858` with 100 fold CV

**Note:** The saved model is an untrained model.

