"""
AgriAI global configuration.
"""

import os
from pathlib import Path


class AgriAISettings:
    PACKAGE_DIR: Path = Path(__file__).parent.parent.resolve()
    BASE_DIR: Path = PACKAGE_DIR.parent.resolve()
    DATA_DIR: Path = BASE_DIR / "data" / "master_datasets"

    OUTPUT_DIR: Path = PACKAGE_DIR / "outputs"
    PREDICTIONS_DIR: Path = OUTPUT_DIR / "predictions"
    RECKONER_DIR: Path = OUTPUT_DIR / "ready_reckoners"
    REPORTS_DIR: Path = OUTPUT_DIR / "reports"

    LOG_DIR: Path = BASE_DIR / "logs"
    CONTRACTS_DIR: Path = BASE_DIR / "contracts"
    CHECKPOINT_DIR: Path = BASE_DIR / ".checkpoints"

    MODELS_DIR: Path = PACKAGE_DIR / "models"
    XGBOOST_DIR: Path = MODELS_DIR / "xgboost"
    REGRESSION_DIR: Path = MODELS_DIR / "regression"
    FUZZY_MODELS_DIR: Path = MODELS_DIR / "fuzzy"

    MEMORY_DIR: Path = PACKAGE_DIR / "memory"
    PAPER_INDEX_DB: Path = MEMORY_DIR / "paper_index.db"
    SCHEMA_HISTORY_DB: Path = MEMORY_DIR / "schema_history.db"
    MODEL_VERSIONS_DIR: Path = MEMORY_DIR / "model_versions"

    RULES_DIR: Path = PACKAGE_DIR / "rules"

    AGENT_RETRY_MAX: int = int(os.getenv("AGRI_AGENT_RETRY_MAX", "3"))
    AGENT_RETRY_DELAY_SEC: float = float(os.getenv("AGRI_AGENT_RETRY_DELAY_SEC", "2.0"))
    AGENT_TIMEOUT_SEC: int = int(os.getenv("AGRI_AGENT_TIMEOUT_SEC", "300"))

    LOG_LEVEL: str = os.getenv("AGRI_LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    EXPORT_FORMATS: list[str] = ["csv", "parquet", "xlsx", "html"]

    BENCHMARK_DIR: Path = OUTPUT_DIR / "benchmark"
    EXPLAINABILITY_DIR: Path = REPORTS_DIR / "explainability"

    CHECKPOINT_ENABLED: bool = True
    INCREMENTAL_MODE: bool = False

    def ensure_dirs(self):
        for d in [self.OUTPUT_DIR, self.PREDICTIONS_DIR, self.RECKONER_DIR,
                  self.REPORTS_DIR, self.LOG_DIR, self.CONTRACTS_DIR,
                  self.CHECKPOINT_DIR, self.MODELS_DIR, self.XGBOOST_DIR,
                  self.REGRESSION_DIR, self.FUZZY_MODELS_DIR,
                  self.MEMORY_DIR, self.MODEL_VERSIONS_DIR,
                  self.BENCHMARK_DIR, self.EXPLAINABILITY_DIR]:
            d.mkdir(parents=True, exist_ok=True)
