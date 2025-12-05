# Data Assets Overview

Paths are relative to the repo root. Generation commands assume the repo is
installed in editable mode (`pip install -e .`) and run from the repo root with
`python -m ...`.

## Raw inputs
- `data/raw/tennis_atp-master/atp_matches_<year>.csv` — Jeff Sackmann match data.
- `data/raw/tennis-data/{year}.xls|xlsx` — Bookmaker odds (tennis-data.co.uk).

## Cleaned base
- `data/cleaned/atp_matches_cleaned.parquet`
  - Built by: `python -m atp_forecaster.data.clean.clean_data`
  - Cleaned/filtered matches; anonymized A/B sides; `result` is target.

## Feature bases
- `data/features/base/player_performance.parquet` — EWMA serve/return stats.
- `data/features/base/glicko2_ratings.parquet` — Pre-match Glicko-2 ratings.
- `data/features/base/experience.parquet` — Match counts overall/surface.
- `data/features/base/fatigue.parquet` — Recent matches/minutes.
- `data/features/base/head_to_head.parquet` — H2H win% and counts.
- `data/features/base/momentum.parquet` — Form delta and Elo momentum.
- Built by the corresponding modules under `atp_forecaster.data.features.*`.

## Joined feature set
- `data/features/feature_sets/dataset_v1_combined.parquet`
  - Built by: `python -m atp_forecaster.data.full.build_dataset_v1`
  - Joins cleaned base + feature bases; columns deduped.

## Training dataset
- `data/training_data/dataset_v1.parquet`
  - Built by: `python -m atp_forecaster.data.full.build_dataset_v1`
  - One-hot encoded categorical fields; matchup diff/log-diff features; `result`
    is the target (last column).

## Backtest dataset
- `data/backtest/backtest_v1.parquet`
  - Built by: `python -m atp_forecaster.data.backtest.build_backtest_v1`
  - Merges odds (tennis-data) with Sackmann features, aligns on names/dates,
    keeps primary odds (`B365W/B365L/PSW/PSL`), engineered matchup features, and
    `result` as target.

