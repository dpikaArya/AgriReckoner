import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "apis.yaml"


def load_apis_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _CONFIG_PATH
    if not path.exists():
        return {"sources": {}, "storage": {}, "pipeline": {}}
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def get_source_config(source_name: str) -> dict[str, Any]:
    cfg = load_apis_config()
    sources = cfg.get("sources", {})
    raw = sources.get(source_name, {})
    merged = dict(raw)
    for key in list(raw.keys()):
        env_key = f"AGRI_{source_name.upper()}_{key.upper()}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            merged[key] = env_val
    return merged


def is_source_enabled(source_name: str) -> bool:
    cfg = get_source_config(source_name)
    return cfg.get("enabled", True)


def get_storage_config() -> dict[str, Any]:
    cfg = load_apis_config()
    return cfg.get("storage", {})


def get_pipeline_config() -> dict[str, Any]:
    cfg = load_apis_config()
    return cfg.get("pipeline", {})
