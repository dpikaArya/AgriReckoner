from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


class XGBoostModel:
    def __init__(self, **kwargs):
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        defaults = dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                        random_state=42, verbosity=0, n_jobs=-1)
        defaults.update(kwargs)
        self.model = XGBRegressor(**defaults)
        self.metrics: dict = {}
        self.feature_names: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])

    def save(self, path: Path) -> Path:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "XGBoostModel":
        import joblib
        return joblib.load(path)

    def feature_importance(self) -> dict[str, float]:
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names,
                            map(float, self.model.feature_importances_)))
        return {}
