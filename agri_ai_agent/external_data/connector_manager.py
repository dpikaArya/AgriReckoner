import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.data_enricher import enrich_master
from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.external_data.registry import ConnectorRegistry
from agri_ai_agent.external_data.registry_db import DatasetRegistry
from src.data_sources.http_client import HttpClient
from src.utils.logging_config import log_api_call

logger = logging.getLogger(__name__)


@dataclass
class ConnectorHealth:
    source_name: str
    is_healthy: bool
    latency_ms: float = 0.0
    discovered_count: int = 0
    error: str | None = None


@dataclass
class ConnectorRunLog:
    source_name: str
    started_at: str
    completed_at: str
    duration_sec: float
    datasets_found: int
    datasets_downloaded: int
    datasets_valid: int
    errors: list[str] = field(default_factory=list)
    success: bool = True


class ConnectorManager:
    def __init__(
        self,
        registry: DatasetRegistry,
        download_dir: Path,
        max_workers: int = 4,
        retry_max_attempts: int = 3,
        retry_base_delay: float = 2.0,
        log_dir: Path | None = None,
    ):
        self._registry = registry
        self._download_dir = download_dir
        self._max_workers = max_workers
        self._retry_max_attempts = retry_max_attempts
        self._retry_base_delay = retry_base_delay
        self._log_dir = log_dir or (download_dir / ".logs")
        self._log_dir.mkdir(parents=True, exist_ok=True)

        ConnectorRegistry.set_registry(registry)
        ConnectorRegistry.discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_sources(self) -> list[str]:
        return ConnectorRegistry.list_sources()

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def health_check(self, source_name: str) -> ConnectorHealth:
        connector = ConnectorRegistry.instantiate(source_name)
        if connector is None:
            return ConnectorHealth(
                source_name=source_name,
                is_healthy=False,
                error="Connector class not found",
            )
        start = time.perf_counter()
        try:
            ok = connector.connect()
            latency = (time.perf_counter() - start) * 1000
            discovered = len(connector.discover()) if ok else 0
            connector.close()
            return ConnectorHealth(
                source_name=source_name,
                is_healthy=ok,
                latency_ms=round(latency, 1),
                discovered_count=discovered,
                error=None if ok else "connect() returned False",
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            connector.close()
            return ConnectorHealth(
                source_name=source_name,
                is_healthy=False,
                latency_ms=round(latency, 1),
                error=str(exc),
            )

    def health_checks(self, sources: list[str] | None = None) -> dict[str, ConnectorHealth]:
        selected = sources if sources is not None else self.list_sources()
        if not selected:
            return {}
        results: dict[str, ConnectorHealth] = {}
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            fut_map = {pool.submit(self.health_check, s): s for s in selected}
            for fut in as_completed(fut_map):
                src = fut_map[fut]
                try:
                    results[src] = fut.result()
                except Exception as exc:
                    results[src] = ConnectorHealth(
                        source_name=src,
                        is_healthy=False,
                        error=str(exc),
                    )
        return results

    # ------------------------------------------------------------------
    # Run a single connector (blocking)
    # ------------------------------------------------------------------

    def run_connector(
        self,
        source_name: str,
    ) -> tuple[str, list[DatasetPackage], ConnectorRunLog]:
        start_time = datetime.now()
        started_at = start_time.isoformat()
        errors: list[str] = []
        packages: list[DatasetPackage] = []

        connector = ConnectorRegistry.instantiate(source_name)
        if connector is None:
            completed_at = datetime.now().isoformat()
            duration = (datetime.now() - start_time).total_seconds()
            log_entry = ConnectorRunLog(
                source_name=source_name,
                started_at=started_at,
                completed_at=completed_at,
                duration_sec=round(duration, 2),
                datasets_found=0,
                datasets_downloaded=0,
                datasets_valid=0,
                errors=["Connector class not found"],
                success=False,
            )
            self._write_run_log(log_entry)
            return source_name, packages, log_entry

        try:
            connected = connector.connect()
            if not connected:
                errors.append("connect() returned False")
                completed_at = datetime.now().isoformat()
                duration = (datetime.now() - start_time).total_seconds()
                log_entry = ConnectorRunLog(
                    source_name=source_name,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_sec=round(duration, 2),
                    datasets_found=0,
                    datasets_downloaded=0,
                    datasets_valid=0,
                    errors=errors,
                    success=False,
                )
                self._write_run_log(log_entry)
                return source_name, packages, log_entry

            datasets = connector.discover()
            found_count = len(datasets)
            logger.info("[%s] discovered %d dataset(s)", source_name, found_count)

            for ds in datasets:
                resource_id = ds.get("id", "")
                if not resource_id:
                    continue

                if self._registry is not None:
                    existing = self._registry.find(source_name, resource_id)
                    if existing and existing.get("status") == "active":
                        newer = self._registry.check_update(
                            source_name,
                            resource_id,
                            existing["version"],
                        )
                        if newer is None:
                            self._registry.update_timestamp(existing["dataset_id"])
                            continue

                dl_dir = self._download_dir / source_name
                dl_dir.mkdir(parents=True, exist_ok=True)

                try:
                    path = self._download_with_retry(connector, resource_id, dl_dir)
                except Exception as exc:
                    errors.append(f"{resource_id}: download failed — {exc}")
                    continue

                if path is None:
                    errors.append(f"{resource_id}: download returned no file")
                    continue

                package = DatasetPackage(
                    source=source_name,
                    resource_id=resource_id,
                    name=path.stem,
                    download_path=path,
                )
                package.data = package.to_dataframe()
                package.is_valid = connector.validate(package)

                if package.is_valid:
                    connector.register(package)
                    packages.append(package)
                else:
                    ve = (
                        "; ".join(package.validation_errors)
                        if package.validation_errors
                        else "validation failed"
                    )
                    errors.append(f"{resource_id}: {ve}")

            connector.close()

        except Exception as exc:
            errors.append(f"unexpected error: {exc}")
            logger.exception("[%s] connector run failed", source_name)
            try:
                connector.close()
            except Exception as e:
                logger.debug("[%s] connector close during error handling: %s", source_name, e)

        completed_at = datetime.now().isoformat()
        duration = (datetime.now() - start_time).total_seconds()
        downloaded = len(packages)
        valid = sum(1 for p in packages if p.is_valid)
        log_entry = ConnectorRunLog(
            source_name=source_name,
            started_at=started_at,
            completed_at=completed_at,
            duration_sec=round(duration, 2),
            datasets_found=found_count,
            datasets_downloaded=downloaded,
            datasets_valid=valid,
            errors=errors,
            success=len(errors) == 0,
        )
        self._write_run_log(log_entry)
        return source_name, packages, log_entry

    # ------------------------------------------------------------------
    # Run all connectors concurrently
    # ------------------------------------------------------------------

    def run_all(
        self,
        sources: list[str] | None = None,
    ) -> tuple[dict[str, list[DatasetPackage]], list[ConnectorRunLog]]:
        selected = sources if sources is not None else self.list_sources()
        if not selected:
            logger.warning("No connectors to run")
            return {}, []

        results: dict[str, list[DatasetPackage]] = {}
        logs: list[ConnectorRunLog] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            fut_map = {pool.submit(self.run_connector, s): s for s in selected}
            for fut in as_completed(fut_map):
                src = fut_map[fut]
                try:
                    _, packages, log_entry = fut.result()
                    results[src] = packages
                    logs.append(log_entry)
                except Exception as exc:
                    logger.error("[%s] connector run crashed: %s", src, exc)
                    results[src] = []
                    logs.append(
                        ConnectorRunLog(
                            source_name=src,
                            started_at=datetime.now().isoformat(),
                            completed_at=datetime.now().isoformat(),
                            duration_sec=0.0,
                            datasets_found=0,
                            datasets_downloaded=0,
                            datasets_valid=0,
                            errors=[f"thread crashed: {exc}"],
                            success=False,
                        )
                    )

        return results, logs

    # ------------------------------------------------------------------
    # Update detection
    # ------------------------------------------------------------------

    def detect_updates(self, source_name: str) -> list[dict]:
        records = self._registry.list_datasets(provider=source_name, status="active")
        updates: list[dict] = []
        for rec in records:
            rid = rec.get("resource_id", "")
            ver = rec.get("version", "1.0.0")
            newer = self._registry.check_update(source_name, rid, ver)
            if newer is not None:
                updates.append(
                    {
                        "resource_id": rid,
                        "current_version": ver,
                        "available_version": newer,
                    }
                )
        return updates

    def detect_all_updates(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for src in self.list_sources():
            updates = self.detect_updates(src)
            if updates:
                result[src] = updates
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_http_client(self, connector: ExternalDataConnector) -> HttpClient:
        api_key = getattr(connector, "api_key", None)
        return HttpClient(
            base_url=connector.base_url,
            api_key=api_key,
            rate_limit=10,
            rate_period=60,
            cache_dir=self._download_dir / ".http_cache",
            default_timeout=30,
            max_retries=self._retry_max_attempts,
        )

    def _download_with_retry(
        self,
        connector: ExternalDataConnector,
        resource_id: str,
        target_dir: Path,
    ) -> Path | None:
        self._build_http_client(connector)
        last_exc: Exception | None = None
        for attempt in range(1, self._retry_max_attempts + 1):
            try:
                path = connector.download(resource_id, target_dir)
                if path is not None and Path(path).exists():
                    log_api_call(
                        logger=logger,
                        method="GET",
                        url=f"{connector.base_url}/download/{resource_id}",
                        status=200,
                        duration_ms=0.0,
                        retries=attempt - 1,
                    )
                    return Path(path)
                if attempt < self._retry_max_attempts:
                    delay = self._retry_base_delay * (2 ** (attempt - 1))
                    logger.info(
                        "[%s] retry %d/%d for %s in %.1fs",
                        connector.source_name,
                        attempt,
                        self._retry_max_attempts,
                        resource_id,
                        delay,
                    )
                    time.sleep(delay)
            except Exception as exc:
                last_exc = exc
                if attempt < self._retry_max_attempts:
                    delay = self._retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "[%s] attempt %d/%d for %s failed: %s — retrying in %.1fs",
                        connector.source_name,
                        attempt,
                        self._retry_max_attempts,
                        resource_id,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    log_api_call(
                        logger=logger,
                        method="GET",
                        url=f"{connector.base_url}/download/{resource_id}",
                        status=0,
                        duration_ms=0.0,
                        retries=attempt - 1,
                    )
                    logger.error(
                        "[%s] all %d attempts failed for %s: %s",
                        connector.source_name,
                        self._retry_max_attempts,
                        resource_id,
                        exc,
                    )
        if last_exc:
            raise last_exc
        return None

    def _write_run_log(self, entry: ConnectorRunLog):
        log_file = self._log_dir / "connector_runs.jsonl"
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry)) + "\n")
        except OSError as e:
            logger.debug("Could not write run log: %s", e)

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_runs(logs: list[ConnectorRunLog]) -> dict:
        total = len(logs)
        succeeded = sum(1 for log in logs if log.success)
        failed = total - succeeded
        total_downloaded = sum(log.datasets_downloaded for log in logs)
        total_valid = sum(log.datasets_valid for log in logs)
        total_duration = sum(log.duration_sec for log in logs)
        return {
            "total_connectors": total,
            "succeeded": succeeded,
            "failed": failed,
            "total_datasets_downloaded": total_downloaded,
            "total_datasets_valid": total_valid,
            "total_duration_sec": round(total_duration, 2),
        }

    @staticmethod
    def enrich_packages(
        packages_by_source: dict[str, list[DatasetPackage]],
        master_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Enrich a master DataFrame by joining external data on spatial/crop keys,
        rather than just appending rows."""
        return enrich_master(master_df, packages_by_source)

    @staticmethod
    def merge_packages(
        packages_by_source: dict[str, list[DatasetPackage]],
        existing_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if existing_df is not None and not existing_df.empty:
            frames.append(existing_df)
        for _, pkgs in packages_by_source.items():
            for pkg in pkgs:
                df = pkg.to_dataframe()
                if df is not None and not df.empty:
                    for col in ["source", "resource_id", "dataset_version", "download_timestamp"]:
                        if col not in df.columns:
                            df[col] = ""
                    df["source"] = pkg.source
                    df["resource_id"] = pkg.resource_id
                    df["dataset_version"] = pkg.version
                    df["download_timestamp"] = datetime.now().isoformat()
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True, sort=False)
        result = result.loc[:, ~result.columns.duplicated(keep="first")]
        return result
