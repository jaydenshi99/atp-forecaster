from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def build_model(
    learning_rate: float = 0.05,
    max_depth: int = 6,
    n_estimators: int = 300,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    min_child_weight: int = 1,
    n_jobs: int = -1,
):
    """Build an XGBoost model."""
    model = Pipeline(
        steps=[
            (
                "clf",
                XGBClassifier(
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_weight=min_child_weight,
                    objective="binary:logistic",
                    tree_method="hist",
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
    return model
