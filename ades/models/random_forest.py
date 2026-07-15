from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class RandomForestModel:
    def __init__(self, **kwargs):
        defaults = dict(n_estimators=100, max_depth=10,
                        random_state=42, n_jobs=-1)
        defaults.update(kwargs)
        self.model = RandomForestRegressor(**defaults)
        self.metrics: dict = {}
        self.feature_names: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
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
    def load(path: Path) -> "RandomForestModel":
        import joblib
        return joblib.load(path)

    def feature_importance(self) -> dict[str, float]:
        return dict(zip(self.feature_names,
                        map(float, self.model.feature_importances_)))
