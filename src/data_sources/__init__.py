import importlib
import inspect
import pkgutil
from typing import Optional

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.registry import ConnectorRegistry

_DATA_SOURCES_PKG = __package__

_connectors: dict[str, type[ExternalDataConnector]] = {}
_initialized = False


def discover(package_path: Optional[str] = None) -> dict[str, type[ExternalDataConnector]]:
    global _initialized
    if _initialized:
        return _connectors

    existing = ConnectorRegistry.discover(
        "agri_ai_agent.external_data.connectors"
    )
    _connectors.update(existing)

    base = package_path or _DATA_SOURCES_PKG
    module = importlib.import_module(base)
    for _, modname, _ in pkgutil.walk_packages(
        path=getattr(module, "__path__", []),
        prefix=base + ".", onerror=lambda x: None,
    ):
        try:
            mod = importlib.import_module(modname)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (name != "ExternalDataConnector"
                        and issubclass(obj, ExternalDataConnector)
                        and not inspect.isabstract(obj)):
                    instance = obj()
                    _connectors[instance.source_name] = obj
        except Exception:
            continue
    _initialized = True
    return _connectors


def list_sources() -> list[str]:
    discover()
    return sorted(_connectors.keys())


def get(source_name: str) -> Optional[type[ExternalDataConnector]]:
    discover()
    return _connectors.get(source_name)


def instantiate(source_name: str) -> Optional[ExternalDataConnector]:
    cls = get(source_name)
    if cls is None:
        return None
    return cls()


__all__ = [
    "discover",
    "list_sources",
    "get",
    "instantiate",
    "ExternalDataConnector",
]
