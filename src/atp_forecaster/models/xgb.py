from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def build_model(
    learning_rate: float = 0.05,
    max_depth: int = 6,
    n_estimators: int = 300,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    min_child_weight: int = 1,
):
    """ build an XGBoost model """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                objective="binary:logistic",
                tree_method="hist",
            )),
        ]
    )
    return model
