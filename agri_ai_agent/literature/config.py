"""Configuration and credential detection for the literature module.

Credentials are read from environment variables and ``config/apis.yaml``.
Each connector declares whether authentication is required and which
environment variables it needs.  If a connector requires credentials and none
are present, the connector is *disabled* — the rest of the pipeline continues.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agri_ai_agent.literature.external_config import (
    load_env_file,
    load_literature_config,
)

logger = logging.getLogger(__name__)

# Environment-prefix convention mirroring the rest of AAIF:
#   AGRI_{SOURCE}_{KEY}
# e.g. AGRI_WOS_API_KEY, AGRI_SCOPUS_API_KEY, AGRI_NASS_API_KEY.
#
# Individual connector modules declare their own required env vars via
# ConnectorAuth; this registry documents every credential the module can
# consume so users can populate a single .env.
CREDENTIAL_REGISTRY: dict[str, tuple[str, ...]] = {
    # ---- literature connectors that require credentials ----
    "Web_of_Science": ("WOS_API_KEY", "AGRI_WOS_API_KEY"),
    "Scopus": ("SCOPUS_API_KEY", "AGRI_SCOPUS_API_KEY"),
    "Dimensions": ("DIMENSIONS_TOKEN", "AGRI_DIMENSIONS_TOKEN"),
    "Mendeley_Data": (
        "MENDELEY_CLIENT_ID",
        "AGRI_MENDELEY_CLIENT_ID",
        "MENDELEY_CLIENT_SECRET",
        "AGRI_MENDELEY_CLIENT_SECRET",
    ),
    "PubMed": ("NCBI_API_KEY", "AGRI_NCBI_API_KEY"),
    "Semantic_Scholar": ("S2_API_KEY", "AGRI_SEMANTIC_SCHOLAR_API_KEY"),
    "DOAJ": ("DOAJ_API_KEY", "AGRI_DOAJ_API_KEY"),
    # ---- agricultural connectors that require credentials ----
    "USDA_NASS": ("NASS_API_KEY", "AGRI_NASS_API_KEY"),
    "USDA_FAS": ("FAS_API_KEY", "AGRI_FAS_API_KEY"),
    "Google_Earth_Engine": (
        "GEE_SERVICE_ACCOUNT",
        "GEE_PRIVATE_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ),
    "Copernicus_CDS": ("CDSAPI_URL", "CDSAPI_KEY"),
    "ERA5": ("CDSAPI_URL", "CDSAPI_KEY"),
    "AgERA5": ("CDSAPI_URL", "CDSAPI_KEY"),
    "ICRISAT": ("DATAVERSE_API_KEY", "AGRI_DATAVERSE_API_KEY"),
    "CIMMYT": ("DATAVERSE_API_KEY", "AGRI_DATAVERSE_API_KEY"),
}


def _env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass
class ConnectorAuth:
    """Declares the credential surface of a connector.

    ``env_vars`` lists environment variables that provide the credential.
    ``extra_headers`` / ``extra_params`` hold static values set at runtime.
    """

    env_vars: tuple[str, ...] = ()
    required: bool = False  # connector only runs when credentials present
    header: str | None = None  # header name for a key-style credential

    def resolve(self) -> str | None:
        for name in self.env_vars:
            value = _env_value(name)
            if value:
                return value
        return None

    def resolve_env(self, name: str) -> str | None:
        return _env_value(name)

    @property
    def available(self) -> bool:
        return self.resolve() is not None

    def describe(self) -> str:
        return ", ".join(self.env_vars)


@dataclass
class LiteratureConfig:
    """Runtime configuration for the literature module."""

    data_dir: Path = Path("literature_data")
    output_dir: Path = Path("outputs")
    state_db: Path = Path("literature_data/state.sqlite")
    bibliography_db: Path = Path("literature_data/bibliography.db")

    # search / harvesting behaviour
    search_terms: tuple[str, ...] = ()
    max_results_per_query: int = 25
    max_queries_per_connector: int = 8
    incremental_mode: bool = True
    fetch_references: bool = True
    fetch_citations: bool = True
    fetch_related: bool = True
    verify_pdfs: bool = True
    timeout_sec: int = 45
    retry_max: int = 3
    retry_base_delay_sec: float = 1.5
    rate_limit_per_minute: int = 30
    cache_dir: Path = Path("literature_data/cache")

    # validation / quality
    validate_dois_live: bool = True
    doi_validation_source: str = "crossref"  # "crossref" | "datacite" | "none"
    quality_min_score: float = 0.0

    # retrain threshold (incremental retraining gate)
    retrain_min_verified_papers: int = 50
    retrain_min_new_papers: int = 10
    auto_retrain: bool = False
    retrain_command: str | None = None

    # contact email sent as mailto to polite APIs
    contact_email: str | None = None

    # externalized config (config/*.yaml), loaded by from_env()
    config_dir: Path | None = None
    source_overrides: dict[str, dict] = field(default_factory=dict)
    connector_limits: dict[str, dict] = field(default_factory=dict)
    quality_weights: dict[str, float] = field(default_factory=dict)
    variable_to_uams: dict[str, str] = field(default_factory=dict)
    schema_mapping: dict = field(default_factory=dict)
    extraction_rules: dict = field(default_factory=dict)
    training_rules: dict = field(default_factory=dict)
    scheduler: dict = field(default_factory=dict)
    dashboard: dict = field(default_factory=dict)

    def source_enabled(self, source_name: str) -> bool | None:
        """External override for connector enablement, or None when absent."""
        override = self.source_overrides.get(source_name)
        if not override or "enabled" not in override:
            return None
        return bool(override["enabled"])

    def limit(self, source_name: str, key: str, default: Any) -> Any:
        """Per-connector limit override (e.g. rate_limit_per_minute)."""
        limits = self.connector_limits.get(source_name) or {}
        return limits.get(key, default)

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> LiteratureConfig:
        root = Path(root) if root else Path.cwd()

        # 1) external credentials + policy files (config/api_keys.env, config/*.yaml)
        load_env_file()
        ext = load_literature_config(root)

        data_dir = Path(os.getenv("AGRI_LIT_DATA_DIR", "literature_data"))
        if not data_dir.is_absolute():
            data_dir = root / data_dir
        output_dir = Path(os.getenv("AGRI_LIT_OUTPUT_DIR", "outputs"))
        if not output_dir.is_absolute():
            output_dir = root / output_dir

        quality = ext["quality_thresholds"]
        cfg = cls(
            data_dir=data_dir,
            output_dir=output_dir,
            state_db=(
                Path(os.getenv("AGRI_LIT_STATE_DB", str(data_dir / "state.sqlite")))
            ),
            bibliography_db=(
                Path(
                    os.getenv(
                        "AGRI_LIT_BIBLIOGRAPHY_DB", str(data_dir / "bibliography.db")
                    )
                )
            ),
            cache_dir=Path(os.getenv("AGRI_LIT_CACHE_DIR", str(data_dir / "cache"))),
            max_results_per_query=int(
                os.getenv("AGRI_LIT_MAX_RESULTS_PER_QUERY", "25")
            ),
            max_queries_per_connector=int(
                os.getenv("AGRI_LIT_MAX_QUERIES_PER_CONNECTOR", "8")
            ),
            incremental_mode=os.getenv("AGRI_LIT_INCREMENTAL", "1") not in {"0", "false", "False"},
            fetch_references=os.getenv("AGRI_LIT_FETCH_REFERENCES", "1")
            not in {"0", "false", "False"},
            fetch_citations=os.getenv("AGRI_LIT_FETCH_CITATIONS", "1")
            not in {"0", "false", "False"},
            fetch_related=os.getenv("AGRI_LIT_FETCH_RELATED", "1")
            not in {"0", "false", "False"},
            verify_pdfs=os.getenv("AGRI_LIT_VERIFY_PDFS", "1")
            not in {"0", "false", "False"},
            timeout_sec=int(os.getenv("AGRI_LIT_TIMEOUT_SEC", "45")),
            retry_max=int(os.getenv("AGRI_LIT_RETRY_MAX", "3")),
            retry_base_delay_sec=float(os.getenv("AGRI_LIT_RETRY_BASE_DELAY", "1.5")),
            rate_limit_per_minute=int(os.getenv("AGRI_LIT_RATE_LIMIT", "30")),
            validate_dois_live=os.getenv("AGRI_LIT_VALIDATE_DOI_LIVE", "1")
            not in {"0", "false", "False"},
            doi_validation_source=os.getenv(
                "AGRI_LIT_DOI_VALIDATION_SOURCE",
                quality.get("doi_validation", {}).get("source", "crossref"),
            ),
            quality_min_score=float(
                os.getenv(
                    "AGRI_LIT_QUALITY_MIN_SCORE",
                    str(quality.get("verified_original", {}).get("min_quality_score", 0)),
                )
            ),
            retrain_min_verified_papers=int(
                os.getenv("AGRI_LIT_RETRAIN_MIN_VERIFIED", "50")
            ),
            retrain_min_new_papers=int(os.getenv("AGRI_LIT_RETRAIN_MIN_NEW", "10")),
            auto_retrain=os.getenv("AGRI_LIT_AUTO_RETRAIN", "0") not in {"0", "false", "False"},
            retrain_command=os.getenv("AGRI_LIT_RETRAIN_COMMAND"),
            contact_email=os.getenv("AGRI_CONTACT_EMAIL"),
            config_dir=ext.get("config_dir"),
            source_overrides=ext["sources"],
            connector_limits=ext["connector_limits"],
            quality_weights=ext["quality_weights"],
            variable_to_uams=ext["variable_to_uams"],
            schema_mapping=ext["schema_mapping"],
            extraction_rules=ext["extraction_rules"],
            training_rules=ext["training_rules"],
            scheduler=ext["scheduler"],
            dashboard=ext["dashboard"],
        )
        if os.getenv("AGRI_LIT_SEARCH_TERMS"):
            cfg.search_terms = tuple(
                t.strip()
                for t in os.getenv("AGRI_LIT_SEARCH_TERMS", "").split(",")
                if t.strip()
            )
        cfg.ensure_dirs()
        return cfg

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.output_dir, self.cache_dir, self.state_db.parent):
            path.mkdir(parents=True, exist_ok=True)


def credential_summary() -> dict[str, bool]:
    """Map connector -> whether its required credentials are present."""
    summary: dict[str, bool] = {}
    for connector, vars_ in CREDENTIAL_REGISTRY.items():
        summary[connector] = any(_env_value(v) for v in vars_)
    return summary
