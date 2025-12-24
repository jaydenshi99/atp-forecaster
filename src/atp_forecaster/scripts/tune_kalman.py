import logging
import joblib
from pathlib import Path

import optuna
import pandas as pd

from atp_forecaster.data.clean import get_cleaned_atp_matches
from atp_forecaster.models.kalman_filter_v1 import KalmanFilterV1
from atp_forecaster.models.kalman_filter_v2 import KalmanFilterV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def suggest_params_v2(trial):
    """Suggest hyperparameters for Optuna trial."""
    return {
        "phi": trial.suggest_float("phi", 0.99, 0.999, log=False),
        "q_g": trial.suggest_float("q_g", 1e-5, 1e-2, log=True),
        "q_d": trial.suggest_float("q_d", 1e-6, 1e-3, log=True),
        "k_elo": trial.suggest_float("k_elo", 0.5, 2.0, log=False),
        "eps_R": trial.suggest_float("eps_R", 1e-6, 1e-3, log=True),
        "init_var_g": trial.suggest_float("init_var_g", 0.1, 1.0, log=False),
        "init_var_d": trial.suggest_float("init_var_d", 0.05, 0.5, log=False),
    }

def suggest_params_v1(trial):
    """Suggest hyperparameters for Optuna trial (v1 - simpler model)."""
    return {
        "init_s": trial.suggest_float("init_s", -0.5, 0.5, log=False),
        "init_P": trial.suggest_float("init_P", 0.1, 1.0, log=False),
        "q": trial.suggest_float("q", 1e-5, 1e-2, log=True),
        "k": trial.suggest_float("k", 0.5, 2.0, log=False),
        "eps_R": trial.suggest_float("eps_R", 1e-6, 1e-3, log=True),
    }


def tune_kalman_filter(df, suggest_params, version='v2', n_trials=50):
    """
    Bayesian hyperparameter optimization for Kalman filter using Optuna.
    
    Args:
        df: DataFrame with match data (must be sorted chronologically)
        suggest_params: Function(trial) -> dict of hyperparameters
        n_trials: Number of Optuna trials
    
    Returns:
        best_params, best_score, study
    """
    # Ensure data is sorted chronologically
    df = df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df[df["tourney_date"].notna()].sort_values("tourney_date").reset_index(drop=True)
    
    def objective(trial):
        params = suggest_params(trial)
        
        # Create Kalman filter with hyperparameters
        if version == 'v2':
            kalman = KalmanFilterV2(**params)
        elif version == 'v1':
            kalman = KalmanFilterV1(**params)
        else:
            raise ValueError(f"Invalid version: {version}")
        
        # Generate features (processes chronologically, naturally doing time-series CV)
        df_with_predictions = kalman.generate_kalman_features(df.copy())
        
        # Evaluate to get log loss
        _, log_loss = kalman.evaluate_kalman_filter(df_with_predictions)
        
        return log_loss

    study = optuna.create_study(direction="minimize")
    # Use multiple workers for parallelization (each trial is independent)
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    best_score = study.best_value

    return best_params, best_score, study


def main():
    """Main function to tune Kalman filter model."""
    logger.info("Loading cleaned ATP matches data...")
    df = get_cleaned_atp_matches()
    logger.info(f"Loaded {len(df)} matches")
    version = 'v1'
    
    # Ensure required columns exist
    required_cols = ["id_a", "id_b", "surface", "tourney_date", "result"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info("Starting hyperparameter tuning...")
    # Select appropriate suggest_params function based on version
    if version == 'v1':
        suggest_params = suggest_params_v1
    elif version == 'v2':
        suggest_params = suggest_params_v2
    else:
        raise ValueError(f"Invalid version: {version}")
    
    best_params, best_score, study = tune_kalman_filter(
        df,
        suggest_params=suggest_params,
        n_trials=100,
        version=version,
    )

    # Create model with best hyperparameters
    if version == 'v2':
        best_model = KalmanFilterV2(**best_params)
    elif version == 'v1':
        best_model = KalmanFilterV1(**best_params)
    else:
        raise ValueError(f"Invalid version: {version}")
    
    # Save model to project root models directory
    project_root = Path(__file__).parent.parent.parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"kalman_{version}.pkl"
    
    joblib.dump(best_model, model_path)
    logger.info(f"Saved model (with best hyperparameters) to {model_path}")
    
    print("\n" + "="*50)
    print("Best hyperparameters:")
    print("="*50)
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest CV log loss: {best_score:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
