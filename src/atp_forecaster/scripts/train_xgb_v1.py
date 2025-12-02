import sys
from pathlib import Path

# Add src directory to Python path so imports work
project_root = Path(__file__).parent.parent.parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import logging
import joblib

from atp_forecaster.training.tuning import tune_model
from atp_forecaster.data import load_processed
from atp_forecaster.models.xgb import build_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def suggest_params(trial):
    """Suggest hyperparameters for Optuna trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
    }

def main():
    """Main function to tune XGBoost model."""
    logger.info("Loading processed training data...")
    X, y = load_processed()
    logger.info(f"Loaded {len(X)} samples with {len(X.columns)} features")
    
    logger.info("Starting hyperparameter tuning...")
    best_model, best_params, best_score, study = tune_model(
        X, y,
        suggest_params=suggest_params,
        build_model=build_model,
        n_trials=1,
        cv=5,
    )
    
    # Save model to project root models directory
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "xgb_v1.pkl"
    
    joblib.dump(best_model, model_path)
    logger.info(f"Saved best model to {model_path}")
    
    print("\n" + "="*50)
    print("Best hyperparameters:")
    print("="*50)
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest CV score: {best_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
