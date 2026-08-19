"""Base connector contract for the Literature Intelligence Module.

Every literature connector implements the same ten capabilities:

    search()                 full-text / boolean search
    lookup_doi()             resolve a DOI to a normalized record
    fetch_metadata()         fetch full metadata for a native identifier
    fetch_pdf_location()     discover legally-accessible full-text locations
    fetch_references()       cited works of a record
    fetch_citations()        works citing a record
    fetch_related_articles() related works suggested by the API
    incremental_sync()       resumable, cursor-based incremental harvest
    validate_record()        structural validation of a normalized record

Cross-cutting behaviour is implemented once here:

* authentication detection (connector auto-disables when required
  credentials are missing);
* request throttling, retry/backoff and disk caching via ConnectorHttpClient;
* structured logging;
* local metadata caching + incremental-sync cursors via ConnectorStateStore;
* DOI normalization helpers.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth, LiteratureConfig
from agri_ai_agent.literature.http import ConnectorHttpClient
from agri_ai_agent.literature.models import (
    LiteratureRecord,
    PdfLocation,
    SyncResult,
)
from agri_ai_agent.literature.state import ConnectorStateStore

logger = logging.getLogger(__name__)

DOI_RE = re.compile(r"^\s*10\.\d{4,9}/[^\s]+$")


def normalize_doi(doi: str | None) -> str | None:
    """Normalize a DOI to its bare lower-case form, or None if invalid."""
    if not doi:
        return None
    doi = doi.strip().lower()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]
    elif doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/") :]
    elif doi.startswith("doi:"):
        doi = doi[4:]
    if not DOI_RE.match(doi):
        return None
    return doi


def validate_doi_syntax(doi: str | None) -> bool:
    return normalize_doi(doi) is not None


def extract_dois(text: str | None) -> list[str]:
    """Extract all DOIs present in a free-text field (references etc.)."""
    if not text:
        return []
    out: list[str] = []
    for token in re.findall(r"(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/[^\s\"'<>]+", text):
        token = token.rstrip(".,);]}")
        doi = normalize_doi(token)
        if doi and doi not in out:
            out.append(doi)
    return out


class LiteratureConnector(ABC):
    # --- subclass contract (metadata) ----------------------------------- #
    source_name: str = ""
    display_name: str = ""
    base_url: str = ""
    contact_email: str | None = None
    # default throttle: polite default, overridden by config
    default_rate_per_minute: int = 30

    auth: ConnectorAuth = ConnectorAuth()

    def __init__(
        self,
        config: LiteratureConfig,
        state: ConnectorStateStore,
        *,
        cache_dir: str | Path | None = None,
    ):
        self.config = config
        self.state = state
        self.logger = logging.getLogger(f"literature.{self.source_name or self.__class__.__name__}")
        cache_dir = cache_dir or config.cache_dir
        source_name = self.source_name or self.__class__.__name__
        self.http = ConnectorHttpClient(
            base_url=self.base_url,
            api_key=None,
            headers=self._auth_headers(),
            rate_limit_per_minute=config.limit(
                source_name,
                "rate_limit_per_minute",
                config.rate_limit_per_minute or self.default_rate_per_minute,
            ),
            timeout_sec=config.limit(source_name, "timeout_sec", config.timeout_sec),
            max_retries=config.limit(source_name, "retry_max", config.retry_max),
            cache_dir=cache_dir,
        )
        self.enabled = self._detect_enabled()
        if not self.enabled:
            self.logger.warning(
                "Connector %s disabled: required credentials missing (%s)",
                self.source_name or self.__class__.__name__,
                self.auth.describe(),
            )

    # ------------------------------------------------------------------ #
    # auth
    # ------------------------------------------------------------------ #
    def _auth_headers(self) -> dict[str, str]:
        if self.auth.header and self.auth.available:
            return {self.auth.header: self.auth.resolve() or ""}
        return {}

    def _detect_enabled(self) -> bool:
        override = self.config.source_enabled(self.source_name)
        if override is not None:
            return override
        if not self.auth.required:
            return True
        return self.auth.available

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        """Search the provider; return normalized records."""

    @abstractmethod
    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        """Resolve a DOI to a normalized record (None when unknown)."""

    @abstractmethod
    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        """Fetch full metadata for a native identifier."""

    @abstractmethod
    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        """Discover legally-accessible full-text locations for a record."""

    @abstractmethod
    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        """Return identifiers (DOIs/native ids) of works cited by ``record``."""

    @abstractmethod
    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        """Return identifiers (DOIs/native ids) of works citing ``record``."""

    @abstractmethod
    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        """Return identifiers of related works suggested by the provider."""

    # ------------------------------------------------------------------ #
    # incremental sync (default implementation)
    # ------------------------------------------------------------------ #
    def incremental_sync(
        self,
        max_records: int | None = None,
        terms: tuple[str, ...] | None = None,
    ) -> SyncResult:
        """Cursor-based incremental harvest.

        Uses the ``search`` capability over the configured boolean query set.
        ``terms`` defaults to ``config.search_terms``.  The last harvested
        offset / timestamp is persisted per connector so later runs continue
        from where they stopped instead of re-fetching everything.
        """
        result = SyncResult(connector=self.source_name)
        if not self.enabled:
            result.errors.append("connector disabled: credentials missing")
            result.completed_at = result.started_at
            return result

        terms = tuple(terms or self.config.search_terms)
        if not terms:
            result.errors.append("no search terms configured")
            result.completed_at = result.started_at
            return result

        per_query = max(self.config.max_results_per_query, 1)
        max_queries = self.config.max_queries_per_connector
        budget = max_records if max_records and max_records > 0 else per_query * max_queries

        try:
            for idx, term in enumerate(terms):
                if budget <= 0:
                    break
                if idx >= max_queries:
                    self.logger.info(
                        "Stopping %s sync: reached max_queries=%d", self.source_name, max_queries
                    )
                    break
                page = 0
                while budget > 0:
                    records = self.search(term, max_results=per_query, page=page)
                    if not records:
                        break
                    for record in records:
                        if budget <= 0:
                            break
                        result.fetched_records += 1
                        if self.state.cache_has(self.source_name, record.source_id):
                            result.skipped_records += 1
                        else:
                            self.state.cache_put(
                                self.source_name, record.source_id, record.to_dict()
                            )
                            result.new_records += 1
                        budget -= 1
                    page += 1
                    if len(records) < per_query:
                        break
            self.state.set_cursor(self.source_name, "last_sync_at", result.started_at)
            result.cursor = result.started_at
        except Exception as exc:  # noqa: BLE001 - connector-level isolation
            self.logger.exception("incremental_sync failed for %s", self.source_name)
            result.errors.append(str(exc))
        result.completed_at = result.started_at  # placeholder, overridden by pipeline
        return result

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def validate_record(self, record: LiteratureRecord) -> list[str]:
        """Structural validation of a normalized record.

        Returns a list of human-readable problems (empty == valid enough).
        """
        errors: list[str] = []
        if not record.source_id and not record.doi:
            errors.append("missing native identifier and DOI")
        if not record.title and not record.doi:
            errors.append("missing title and DOI")
        if record.doi and not validate_doi_syntax(record.doi):
            errors.append(f"malformed DOI: {record.doi!r}")
        if record.year is not None and not (1800 <= record.year <= 2100):
            errors.append(f"implausible year: {record.year}")
        return errors

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def get_cursor(self, key: str, default: Any = None) -> Any:
        return self.state.get_cursor(self.source_name, key, default)

    def set_cursor(self, key: str, value: Any) -> None:
        self.state.set_cursor(self.source_name, key, value)

    def cache_record(self, record: LiteratureRecord) -> None:
        self.state.cache_put(self.source_name, record.source_id, record.raw)

    def close(self) -> None:  # noqa: B027 - optional resource release, no-op base
        """Release any connector-level resources (default: no-op)."""
