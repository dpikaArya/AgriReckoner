"""
ADES global configuration.
"""

import os
from pathlib import Path


class ADESSettings:
    BASE_DIR: Path = Path(__file__).parent.parent.parent.resolve()
    DATA_DIR: Path = BASE_DIR / "data" / "master_datasets"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    LOG_DIR: Path = BASE_DIR / "logs"
    REPORT_DIR: Path = BASE_DIR / "reports"
    CONTRACTS_DIR: Path = BASE_DIR / "contracts"
    CHECKPOINT_DIR: Path = BASE_DIR / ".checkpoints"
    MODELS_DIR: Path = BASE_DIR / "outputs" / "models"

    AGENT_RETRY_MAX: int = int(os.getenv("ADES_AGENT_RETRY_MAX", "3"))
    AGENT_RETRY_DELAY_SEC: float = float(os.getenv("ADES_AGENT_RETRY_DELAY_SEC", "2.0"))
    AGENT_TIMEOUT_SEC: int = int(os.getenv("ADES_AGENT_TIMEOUT_SEC", "300"))

    LOG_LEVEL: str = os.getenv("ADES_LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    EXPORT_FORMATS: list[str] = ["csv", "parquet", "sqlite", "duckdb", "xlsx"]

    CHECKPOINT_ENABLED: bool = True
    INCREMENTAL_MODE: bool = False

    ONTOLOGY_SOURCES: list[str] = [
        "AGROVOC", "Crop Ontology", "Plant Ontology",
        "Environment Ontology", "FAO Vocabulary",
    ]

    EVALUATED_MODELS: list[str] = [
        "Multiple Linear Regression", "Polynomial Regression",
        "Ridge Regression", "Lasso Regression", "Elastic Net",
        "Random Forest", "Extra Trees", "XGBoost", "LightGBM",
        "CatBoost", "Support Vector Regression", "KNN",
        "Deep Learning", "Neural Networks", "LSTM",
        "Transformer Models", "Time Series Forecasting",
        "Explainable AI", "Causal Inference",
        "Digital Twins", "Future Agentic AI Systems",
    ]

    def ensure_dirs(self):
        for d in [self.OUTPUT_DIR, self.LOG_DIR, self.REPORT_DIR,
                  self.CONTRACTS_DIR, self.CHECKPOINT_DIR]:
            d.mkdir(parents=True, exist_ok=True)
