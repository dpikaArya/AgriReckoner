"""Mendeley Data connector (api.mendeley.com/data, official OAuth2 API).

Requires a Mendeley Data client id + secret (client_credentials grant).
"""

from __future__ import annotations

import time
from typing import Any

from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.connector import LiteratureConnector, normalize_doi
from agri_ai_agent.literature.models import Author, LiteratureRecord, PdfLocation


class MendeleyDataConnector(LiteratureConnector):
    source_name = "Mendeley_Data"
    display_name = "Mendeley Data"
    base_url = "https://api.mendeley.com"
    default_rate_per_minute = 30

    auth = ConnectorAuth(
        env_vars=(
            "MENDELEY_CLIENT_ID",
            "AGRI_MENDELEY_CLIENT_ID",
            "MENDELEY_CLIENT_SECRET",
            "AGRI_MENDELEY_CLIENT_SECRET",
        ),
        required=True,
    )

    def __init__(self, *args: Any, **kwargs: Any):
        self._token: str | None = None
        self._token_expiry: float = 0.0
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # OAuth2 (client_credentials)
    # ------------------------------------------------------------------ #
    def _obtain_token(self) -> str | None:
        client_id = self.auth.resolve_env("MENDELEY_CLIENT_ID") or self.auth.resolve_env(
            "AGRI_MENDELEY_CLIENT_ID"
        )
        client_secret = self.auth.resolve_env("MENDELEY_CLIENT_SECRET") or self.auth.resolve_env(
            "AGRI_MENDELEY_CLIENT_SECRET"
        )
        if not client_id or not client_secret:
            return None
        try:
            payload = self.http.post_json(
                "/oauth/token",
                payload={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("Mendeley token request failed: %s", exc)
            return None
        if isinstance(payload, dict) and payload.get("access_token"):
            self._token = payload["access_token"]
            self._token_expiry = time.time() + int(payload.get("expires_in", 3600)) - 60
        return self._token

    def _auth_headers(self) -> dict[str, str]:
        token = self._token if self._token and time.time() < self._token_expiry else None
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    def _bearer(self) -> dict[str, str]:
        token = self._token
        if not token or time.time() >= self._token_expiry:
            token = self._obtain_token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    # ------------------------------------------------------------------ #
    # record mapping
    # ------------------------------------------------------------------ #
    def _to_record(self, dataset: dict[str, Any]) -> LiteratureRecord:
        doi = normalize_doi(dataset.get("doi"))
        authors = [
            Author(
                full_name=a.get("name", ""),
                given_name=a.get("given_name"),
                family_name=a.get("surname"),
                orcid=(a.get("orcid") or None),
            )
            for a in dataset.get("authors", []) or []
        ]
        pdf_locations: list[PdfLocation] = []
        for file_ in dataset.get("files", []) or []:
            if (file_.get("name") or "").lower().endswith(".pdf") and file_.get("download_link"):
                pdf_locations.append(
                    PdfLocation(
                        url=file_["download_link"],
                        source_kind="repository",
                        content_type="application/pdf",
                        doi=doi,
                        source_repository="Mendeley Data",
                    )
                )
        return LiteratureRecord(
            source=self.source_name,
            source_id=str(dataset.get("id", "")),
            doi=doi,
            title=dataset.get("title"),
            abstract=dataset.get("description"),
            year=_year(dataset.get("created") or dataset.get("modified")),
            publisher=dataset.get("publisher"),
            authors=authors,
            publication_type="dataset",
            language=dataset.get("language"),
            subjects=dataset.get("subject_areas", []) or [],
            keywords=[k.get("name") for k in dataset.get("keywords", []) or [] if k.get("name")][:20],
            pdf_locations=pdf_locations,
            license=dataset.get("license"),
            landing_page=dataset.get("link"),
            raw=dataset,
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 25, **kwargs: Any) -> list[LiteratureRecord]:
        page = int(kwargs.get("page", 0))
        payload = self.http.get_json(
            "/data/datasets",
            params={"query": query, "limit": max_results, "offset": page * max_results},
            headers=self._bearer(),
        )
        if not isinstance(payload, list):
            return []
        records: list[LiteratureRecord] = []
        for item in payload:
            record = self.fetch_metadata(str(item.get("id", "")))
            if record:
                records.append(record)
            if len(records) >= max_results:
                break
        return records

    def lookup_doi(self, doi: str) -> LiteratureRecord | None:
        payload = self.http.get_json(
            "/data/datasets",
            params={"doi": normalize_doi(doi)},
            headers=self._bearer(),
        )
        if not isinstance(payload, list) or not payload:
            return None
        return self.fetch_metadata(str(payload[0].get("id", "")))

    def fetch_metadata(self, source_id: str) -> LiteratureRecord | None:
        payload = self.http.get_json(f"/data/datasets/{source_id}", headers=self._bearer())
        if not isinstance(payload, dict):
            return None
        files = self.http.get_json(f"/data/datasets/{source_id}/files", headers=self._bearer())
        if isinstance(files, list):
            payload["files"] = files
        return self._to_record(payload)

    def fetch_pdf_location(self, record: LiteratureRecord) -> list[PdfLocation]:
        return record.pdf_locations

    def fetch_references(self, record: LiteratureRecord) -> list[str]:
        return []  # not exposed by the Mendeley Data API

    def fetch_citations(self, record: LiteratureRecord) -> list[str]:
        return []

    def fetch_related_articles(self, record: LiteratureRecord) -> list[str]:
        return []


def _year(iso_date: str | None) -> int | None:
    if not iso_date:
        return None
    try:
        return int(str(iso_date)[:4])
    except (TypeError, ValueError):
        return None
