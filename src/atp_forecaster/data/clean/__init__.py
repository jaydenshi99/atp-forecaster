"""Data loading utilities."""

import pandas as pd
from pathlib import Path


def get_cleaned_atp_matches():
    """
    Load the cleaned ATP matches dataset.
    """
    project_root = Path(__file__).resolve().parents[4]
    atp_matches_path = project_root / "data" / "cleaned" / "atp_matches_cleaned.parquet"
    df = pd.read_parquet(atp_matches_path)
    df = df.sort_values(['order']).reset_index(drop=True)
    return df

def load_processed():
    """
    Load the processed training dataset.
    
    Returns:
        X: Feature matrix (all columns except 'result')
        y: Target vector ('result' column)
    """
    # Get project root (assuming this file is in src/atp_forecaster/data/clean/)
    # parents[4] corresponds to the repository root when this file lives under
    # src/atp_forecaster/data/clean/__init__.py
    project_root = Path(__file__).resolve().parents[4]
    
    dataset_path = project_root / "data" / "training_data" / "dataset_v1.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {dataset_path}. "
            "Please run atp_forecaster.data.full.build_dataset_v1 to generate it."
        )
    
    df = pd.read_parquet(dataset_path)
    
    # Separate features and target
    if 'result' not in df.columns:
        raise ValueError("Dataset must contain 'result' column")
    
    X = df.drop(columns=['recent_minutes_diff',
       'log_total_surface_matches_diff',
       'log_recent_matches_diff',
       'result'])
    y = df['result']
    
    return X, y

