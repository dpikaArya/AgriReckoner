"""Manager for agricultural data connectors: discovery, health and sync."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agri_ai_agent.literature.agri.connector import AgriculturalDataConnector
from agri_ai_agent.literature.config import LiteratureConfig
from agri_ai_agent.literature.models import AgriculturalRecord, SyncResult
from agri_ai_agent.literature.state import ConnectorStateStore

logger = logging.getLogger(__name__)


@dataclass
class ConnectorHealth:
    source_name: str
    enabled: bool
    reason: str = ""
    datasets: int = 0


def discover_agricultural_connectors() -> list[type[AgriculturalDataConnector]]:
    """Import every agricultural connector module and return their classes."""
    from agri_ai_agent.literature.agri import sources  # noqa: F401  (registers via __all__)

    classes: list[type[AgriculturalDataConnector]] = []
    for name in dir(sources):
        obj = getattr(sources, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, AgriculturalDataConnector)
            and obj is not AgriculturalDataConnector
            and getattr(obj, "source_name", "")
        ):
            classes.append(obj)
    # dedupe by source_name
    seen: dict[str, type[AgriculturalDataConnector]] = {}
    for cls in classes:
        seen.setdefault(cls.source_name, cls)
    return list(seen.values())


class AgriculturalDataManager:
    def __init__(self, config: LiteratureConfig, state: ConnectorStateStore):
        self.config = config
        self.state = state
        self.connectors: list[AgriculturalDataConnector] = []
        self.health: dict[str, ConnectorHealth] = {}

    def build_connectors(
        self, classes: list[type[AgriculturalDataConnector]] | None = None
    ) -> None:
        for cls in classes or discover_agricultural_connectors():
            try:
                connector = cls(self.config, self.state)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to instantiate agricultural connector %s", cls)
                self.health[cls.source_name] = ConnectorHealth(
                    cls.source_name, False, f"instantiation error: {exc}"
                )
                continue
            self.connectors.append(connector)
            self.health[connector.source_name] = ConnectorHealth(
                connector.source_name,
                connector.enabled,
                "" if connector.enabled else "credentials missing",
            )

    @property
    def enabled_connectors(self) -> list[AgriculturalDataConnector]:
        return [c for c in self.connectors if c.enabled]

    def run_all(self, queries: tuple[str, ...] | None = None) -> list[SyncResult]:
        results: list[SyncResult] = []
        for connector in self.enabled_connectors:
            self.health[connector.source_name].datasets = 0
            result = connector.incremental_sync(queries=queries)
            results.append(result)
            self.health[connector.source_name].datasets = result.fetched_records
            self.state.set_cursor("_agri_run", connector.source_name, result.to_dict())
        return results

    def collect_records(self) -> list[AgriculturalRecord]:
        """Rehydrate all cached agricultural records from the state store."""
        records: list[AgriculturalRecord] = []
        for connector in self.connectors:
            for _source_id, payload in self.state.iter_cache(connector.source_name):
                try:
                    records.append(AgriculturalRecord(**payload))
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "skip unparsable cached record %s/%s: %s",
                        connector.source_name,
                        _source_id,
                        exc,
                    )
        return records
