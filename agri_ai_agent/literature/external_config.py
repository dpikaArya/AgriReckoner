"""Externalized configuration for the Literature Intelligence Module.

All operational settings live in ``config/*.yaml`` plus ``config/api_keys.env``
(the repository root ``config/`` directory, overridable via ``AGRI_CONFIG_DIR``).

* ``load_env_file``   — populate environment variables from a KEY=VALUE file
  (never overwrites a value that is already set).
* ``load_yaml``       — safe YAML load with an empty-dict fallback (used for
  optional/auxiliary files and by callers that must never fail).
* ``load_literature_config`` — read every config file and return one nested
  dict that ``LiteratureConfig.from_env`` merges into the dataclass.  Raises
  :class:`LiteratureConfigError` with an informative message when a required
  config file is missing or unparsable, instead of silently returning ``{}``.

Every provider value in these files points at the official production API;
no placeholder endpoints are shipped.
"""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_FILES = (
    "literature_sources.yaml",
    "connector_limits.yaml",
    "quality_thresholds.yaml",
    "schema_mapping.yaml",
    "extraction_rules.yaml",
    "training_rules.yaml",
    "scheduler.yaml",
    "dashboard.yaml",
)

ENV_FILE = "api_keys.env"


class LiteratureConfigError(RuntimeError):
    """Raised when required externalized literature configuration is unusable.

    Replaces the previous silent ``{}`` fallback so misconfiguration (a missing
    or unparsable ``config/*.yaml``) surfaces with an actionable message.
    """


def config_root() -> Path:
    """Resolve the external config directory (default: repo-root ``config/``)."""
    override = os.getenv("AGRI_CONFIG_DIR")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2] / "config"


def load_env_file(path: str | Path | None = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from a .env file into the process environment.

    Only variables that are not already set are applied, so OS-level and
    shell-provided credentials always win.  Values with surrounding quotes are
    stripped.  ``#`` comment lines and blank lines are skipped.
    """
    env_path = Path(path) if path else config_root() / ENV_FILE
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if os.getenv(key) is None:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def load_yaml(path: str | Path) -> dict:
    """Safe-load a YAML file, returning ``{}`` when missing or unparsable."""
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a runtime dependency
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - config must never crash the pipeline
        return {}


def load_required_yaml(path: str | Path, label: str | None = None) -> dict:
    """Load a YAML config file, raising an informative error when unusable.

    This is the strict counterpart of :func:`load_yaml`: required module
    configuration must parse, so a missing or malformed file is reported with
    the offending path and the root cause instead of being swallowed.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is a runtime dependency
        raise LiteratureConfigError(
            "PyYAML is required to load literature configuration but is not installed."
        ) from exc
    p = Path(path)
    name = label or p.name
    if not p.exists():
        raise LiteratureConfigError(
            f"Missing required literature config file: {name!r} (expected at {p}). "
            f"Point AGRI_CONFIG_DIR at a directory containing all of "
            f"{', '.join(CONFIG_FILES)}."
        )
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:  # noqa: BLE001 - wrap parse failures with context
        raise LiteratureConfigError(
            f"Unparsable literature config file {p} ({exc.__class__.__name__}: {exc})."
        ) from exc
    if not isinstance(data, dict):
        raise LiteratureConfigError(
            f"Literature config file {p} does not contain a YAML mapping "
            f"(found {type(data).__name__})."
        )
    return data


def load_literature_config(root: str | Path | None = None) -> dict:
    """Load every external config file into a single nested dict.

    Returns a dict shaped as::

        {
          "sources": {source_name: {enabled: bool}},
          "connector_limits": {source_name: {rate_limit_per_minute: int, ...}},
          "quality_thresholds": {...},
          "schema_mapping": {...},
          "extraction_rules": {...},
          "training_rules": {...},
          "scheduler": {...},
          "dashboard": {...},
        }
    """
    base = Path(root).resolve() if root else config_root()
    cfg_root = base if (base / "literature_sources.yaml").exists() else config_root()

    sources = load_required_yaml(cfg_root / "literature_sources.yaml")
    limits = load_required_yaml(cfg_root / "connector_limits.yaml")
    quality = load_required_yaml(cfg_root / "quality_thresholds.yaml")
    schema_map = load_required_yaml(cfg_root / "schema_mapping.yaml")
    extraction = load_required_yaml(cfg_root / "extraction_rules.yaml")
    training = load_required_yaml(cfg_root / "training_rules.yaml")
    scheduler = load_required_yaml(cfg_root / "scheduler.yaml")
    dashboard = load_required_yaml(cfg_root / "dashboard.yaml")

    def _merge_sections(data: dict) -> dict:
        """Flatten `{group: {source: {...}}}` into `{source: {...}}`."""
        merged: dict = {}
        for group in ("literature", "agricultural"):
            for source, settings in dict(data.get(group) or {}).items():
                merged[source] = settings
        return merged

    source_overrides = _merge_sections(sources)
    connector_limits = _merge_sections(limits)

    return {
        "config_dir": cfg_root,
        "sources": source_overrides,
        "connector_limits": connector_limits,
        "quality_thresholds": dict(quality.get("quality_thresholds") or {}),
        "quality_weights": dict(
            (quality.get("quality_thresholds") or {}).get("quality_weights") or {}
        ),
        "schema_mapping": dict(schema_map.get("schema_mapping") or {}),
        "variable_to_uams": dict(
            (schema_map.get("schema_mapping") or {}).get("variable_to_uams") or {}
        ),
        "extraction_rules": dict(extraction.get("extraction_rules") or {}),
        "training_rules": dict(training.get("training_rules") or {}),
        "scheduler": dict(scheduler.get("scheduler") or {}),
        "dashboard": dict(dashboard.get("dashboard") or {}),
    }
