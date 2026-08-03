"""Discovery of literature connectors.

Connector modules live under ``agri_ai_agent.literature.connectors``.  Each
module exposes one or more ``LiteratureConnector`` subclasses; this registry
imports every module and returns the connector classes, so the pipeline can
build and run them without hard-coding the list.
"""

from __future__ import annotations

import importlib
import pkgutil

from agri_ai_agent.literature.connector import LiteratureConnector

_PACKAGE = "agri_ai_agent.literature.connectors"


def discover_literature_connectors(
    extra_classes: list[type[LiteratureConnector]] | None = None,
) -> list[type[LiteratureConnector]]:
    """Import connector modules and return their classes (deduplicated)."""
    classes: list[type[LiteratureConnector]] = list(extra_classes or [])
    package = importlib.import_module(_PACKAGE)
    for _info in pkgutil.iter_modules(package.__path__):
        if _info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{_PACKAGE}.{_info.name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, LiteratureConnector)
                and obj is not LiteratureConnector
                and getattr(obj, "source_name", "")
            ):
                classes.append(obj)
    seen: dict[str, type[LiteratureConnector]] = {}
    for cls in classes:
        seen.setdefault(cls.source_name, cls)
    return list(seen.values())
