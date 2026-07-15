from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso


class LinearRegressionModel:
    def __init__(self, model_type: str = "linear", **kwargs):
        if model_type == "ridge":
            params = dict(alpha=kwargs.pop("alpha", 1.0), random_state=kwargs.get("random_state", 42))
            self.model = Ridge(**params)
        elif model_type == "lasso":
            params = dict(alpha=kwargs.pop("alpha", 0.1), random_state=kwargs.get("random_state", 42))
            self.model = Lasso(**params)
        else:
            self.model = LinearRegression()
        self.model_type = model_type
        self.metrics: dict = {}
        self.feature_names: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
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
    def load(path: Path) -> "LinearRegressionModel":
        import joblib
        return joblib.load(path)

    def coefficients(self) -> dict[str, float]:
        coef = self.model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return dict(zip(self.feature_names, map(float, coef)))

    def intercept(self) -> float:
        return float(self.model.intercept_)
