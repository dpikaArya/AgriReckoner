import importlib
import inspect
import pkgutil
from typing import Optional

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.registry_db import DatasetRegistry
from agri_ai_agent.utils.logging_utils import get_logger

_logger = get_logger("ConnectorRegistry")


class ConnectorRegistry:
    _connectors: dict[str, type[ExternalDataConnector]] = {}
    _initialized = False
    _registry: Optional[DatasetRegistry] = None

    @classmethod
    def set_registry(cls, registry: DatasetRegistry):
        cls._registry = registry

    @classmethod
    def discover(cls, package_path: Optional[str] = None) -> dict[str, type[ExternalDataConnector]]:
        if cls._initialized:
            return cls._connectors
        base = package_path or __package__ + ".connectors"
        module = importlib.import_module(base)
        for _, modname, _ in pkgutil.walk_packages(
            path=module.__path__, prefix=base + ".", onerror=lambda x: None
        ):
            try:
                mod = importlib.import_module(modname)
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if (
                        name != "ExternalDataConnector"
                        and issubclass(obj, ExternalDataConnector)
                        and not inspect.isabstract(obj)
                    ):
                        instance = obj()
                        cls._connectors[instance.source_name] = obj
            except Exception as e:
                _logger.debug("Could not register connector from module %s: %s", name, e)
        cls._initialized = True
        return cls._connectors

    @classmethod
    def get(cls, source_name: str) -> Optional[type[ExternalDataConnector]]:
        if not cls._initialized:
            cls.discover()
        return cls._connectors.get(source_name)

    @classmethod
    def list_sources(cls) -> list[str]:
        if not cls._initialized:
            cls.discover()
        return sorted(cls._connectors.keys())

    @classmethod
    def instantiate(cls, source_name: str) -> Optional[ExternalDataConnector]:
        connector_cls = cls.get(source_name)
        if connector_cls is None:
            return None
        instance = connector_cls()
        if cls._registry is not None:
            instance.set_registry(cls._registry)
        return instance
