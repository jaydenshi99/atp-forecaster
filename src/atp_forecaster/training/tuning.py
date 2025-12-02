import numpy as np
import optuna
import logging
from typing import Callable

from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def time_series_cv(X, y, model, n_splits=100, debug=True):
    fold_size = len(X) // (n_splits + 1)
    aucs, accs, losses = [], [], []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        X_train, y_train = X.iloc[:train_end], y[:train_end]
        X_test,  y_test  = X.iloc[train_end:train_end + fold_size], y[train_end:train_end + fold_size]

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:,1]
        
        auc = roc_auc_score(y_test, preds)
        pred_class = (preds >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred_class)
        loss = log_loss(y_test, preds)

        aucs.append(auc)
        accs.append(acc)
        losses.append(loss)

        if debug:
            print(f"{i} AUC={auc:.4f}, Accuracy={acc:.4f}, LogLoss={loss:.4f}")

    if debug:
        print(f"\nAUC:      mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}")
        print(f"Accuracy: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}")
        print(f"LogLoss:  mean={np.mean(losses):.4f}, std={np.std(losses):.4f}")

    return aucs, accs, losses

def tune_model(
    X,
    y,
    suggest_params,
    build_model: Callable[..., object],
    n_trials: int = 50,
    cv: int = 3,
    n_jobs: int = -1,
    direction: str = "maximize",
):
    """
    Bayesian hyperparameter optimisation using Optuna.

    - X, y: training data
    - suggest_params: function(trial) -> dict of kwargs for search range
    - build_model: callable(**params) -> estimator with fit / predict_proba
    """

    def objective(trial: optuna.Trial) -> float:
        clf_kwargs = suggest_params(trial)

        model = build_model(**clf_kwargs)

        aucs, _, _ = time_series_cv(X, y, model, n_splits=cv, debug=False)
        return np.mean(aucs)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    # Rebuild best model + fit on full data
    best_model = build_model(**best_params)
    best_model.fit(X, y)

    return best_model, best_params, best_value, study