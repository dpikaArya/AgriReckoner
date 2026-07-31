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

    RULES_DIR: Path = PACKAGE_DIR / "rules"

    AGENT_RETRY_MAX: int = int(os.getenv("AGRI_AGENT_RETRY_MAX", "3"))
    AGENT_RETRY_DELAY_SEC: float = float(os.getenv("AGRI_AGENT_RETRY_DELAY_SEC", "2.0"))

    LOG_LEVEL: str = os.getenv("AGRI_LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # LLM extraction (optional; the framework runs offline without a key).
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("AGRI_LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("AGRI_LLM_TEMPERATURE", "0.0"))

    BENCHMARK_DIR: Path = OUTPUT_DIR / "benchmark"
    EXPLAINABILITY_DIR: Path = REPORTS_DIR / "explainability"

    EXTERNAL_DATA_DIR: Path = BASE_DIR / "external_data"
    DATASET_REGISTRY_PATH: Path = BASE_DIR / "database" / "dataset_registry.sqlite"

    CONTINUOUS_LEARNING_DB: Path = BASE_DIR / "database" / "continuous_learning.db"
    CONTINUOUS_LEARNING_DIR: Path = OUTPUT_DIR / "continuous_learning"

    CHECKPOINT_ENABLED: bool = True
    INCREMENTAL_MODE: bool = False

    def ensure_dirs(self):
        for d in [
            self.OUTPUT_DIR,
            self.PREDICTIONS_DIR,
            self.RECKONER_DIR,
            self.REPORTS_DIR,
            self.LOG_DIR,
            self.CONTRACTS_DIR,
            self.CHECKPOINT_DIR,
            self.MODELS_DIR,
            self.BENCHMARK_DIR,
            self.EXPLAINABILITY_DIR,
            self.EXTERNAL_DATA_DIR,
            self.CONTINUOUS_LEARNING_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)
