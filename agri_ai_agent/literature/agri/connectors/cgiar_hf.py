"""CGIAR connector via the official Hugging Face datasets hub.

CGIAR data is officially mirrored on Hugging Face (the CGIAR community org);
the HF datasets API is verified reachable and requires no authentication for
public metadata listing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agri_ai_agent.literature.agri.connector import (
    AgriculturalDataConnector,
    DatasetDescriptor,
)
from agri_ai_agent.literature.config import ConnectorAuth
from agri_ai_agent.literature.models import AgriculturalRecord

_HF_API = "https://huggingface.co/api/datasets"
_CGIAR_ORGS = ("CGIAR", "CGIAR-Ag")


class CgiarHuggingFaceConnector(AgriculturalDataConnector):
    source_name = "CGIAR"
    display_name = "CGIAR (Hugging Face mirror)"
    base_url = _HF_API
    default_rate_per_minute = 20

    auth = ConnectorAuth(env_vars=(), required=False)

    def _datasets(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for org in _CGIAR_ORGS:
            payload = self.http.get_json("", params={"author": org, "limit": 100})
            if isinstance(payload, list):
                out.extend(payload)
        return out

    def _descriptor(self, ds: dict[str, Any]) -> DatasetDescriptor:
        ds_id = ds.get("id") or ""
        url = ds.get("url") or f"https://huggingface.co/datasets/{ds_id}"
        return DatasetDescriptor(
            dataset_id=ds_id,
            title=ds_id,
            url=url,
            description=(ds.get("description") or "")[:500],
            license=ds.get("cardData", {}).get("license") if isinstance(ds.get("cardData"), dict) else None,
            variable=None,
            files=[],
            metadata={"downloads": ds.get("downloads"), "likes": ds.get("likes")},
        )

    # ------------------------------------------------------------------ #
    # required interface
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        tokens = query.lower().split()
        hits = [
            d
            for d in self._datasets()
            if not tokens or all(t in d.get("id", "").lower() for t in tokens)
        ]
        return [self._descriptor(d) for d in hits[:max_results]]

    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        payload = self.http.get_json(f"/{dataset_id}", params={"full": "false"})
        if not isinstance(payload, dict):
            return None
        return self._descriptor(payload)

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None:
            return None
        # download the dataset README as provenance metadata
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"cgiar_{dataset_id.replace('/', '_')}.json"
        path.write_text(json.dumps(descriptor.to_dict(), indent=2), encoding="utf-8")
        return path

    def to_records(self, dataset_id: str, download_path: Path | None = None) -> list[AgriculturalRecord]:
        if download_path is None or not download_path.exists():
            return []
        descriptor = json.loads(download_path.read_text(encoding="utf-8"))
        return [
            AgriculturalRecord(
                source=self.source_name,
                dataset_id=dataset_id,
                variable=descriptor.get("variable"),
                value=None,
                unit=None,
                provenance=descriptor.get("url"),
                license=descriptor.get("license"),
                extra={"rows": 1, "title": descriptor.get("title")},
            )
        ]
