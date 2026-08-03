"""Base connector for official Dataverse repositories.

Used by the CGIAR, IFPRI, HarvestChoice, ICRISAT, IRRI, CIMMYT and MapSPAM
connectors.  Implements the standard Dataverse search API and dataset
download endpoints; subclasses supply the instance base URL and the optional
``subtree`` to restrict searches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agri_ai_agent.literature.agri.connector import (
    AgriculturalDataConnector,
    DatasetDescriptor,
)
from agri_ai_agent.literature.models import AgriculturalRecord


class DataverseConnector(AgriculturalDataConnector):
    instance_url: str = ""  # e.g. https://dataverse.harvard.edu
    subtree: str | None = None  # e.g. "IFPRI", "spam"

    def __init__(self, *args, **kwargs):
        self.base_url = self.instance_url.rstrip("/")
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = 20) -> list[DatasetDescriptor]:
        params: dict[str, Any] = {
            "q": query,
            "type": "dataset",
            "per_page": min(max_results, 100),
            "sort": "dateSort",
            "order": "desc",
        }
        if self.subtree:
            params["subtree"] = self.subtree
        payload = self.http.get_json("/api/search", params=params)
        if not isinstance(payload, dict) or payload.get("status") != "OK":
            self.logger.warning("Dataverse search failed for %s", self.source_name)
            return []
        out: list[DatasetDescriptor] = []
        for item in payload.get("data", {}).get("items", []) or []:
            if item.get("type") != "dataset":
                continue
            global_id = item.get("global_id") or item.get("url") or ""
            out.append(
                DatasetDescriptor(
                    dataset_id=global_id or item.get("dataset_id") or item.get("id", ""),
                    title=item.get("name") or item.get("title") or "",
                    url=item.get("url") or "",
                    description=item.get("description") or "",
                    metadata=item,
                    temporal={},
                    spatial={},
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # metadata + download
    # ------------------------------------------------------------------ #
    def fetch_metadata(self, dataset_id: str) -> DatasetDescriptor | None:
        persistent = dataset_id if dataset_id.startswith("doi:") else f"doi:{dataset_id}"
        payload = self.http.get_json(
            "/api/datasets/:persistentId",
            params={"persistentId": persistent},
        )
        if not isinstance(payload, dict) or payload.get("status") != "OK":
            return None
        data = payload.get("data", {})
        title = ""
        for block in data.get("metadataBlocks", {}).values():
            for field in block.get("fields", []) or []:
                if field.get("typeName") == "title":
                    title = field.get("value", "")
        files = [
            {
                "url": f"{self.instance_url}/api/access/datafile/{f.get('id')}",
                "name": f.get("filename"),
                "format": Path(f.get("filename") or "").suffix.lstrip("."),
                "id": f.get("id"),
            }
            for f in data.get("files", []) or []
        ]
        return DatasetDescriptor(
            dataset_id=dataset_id,
            title=title or dataset_id,
            url=f"https://doi.org/{dataset_id[4:]}" if dataset_id.startswith("doi:") else dataset_id,
            description="",
            files=files,
            metadata=data,
        )

    def download(self, dataset_id: str, target_dir: Path) -> Path | None:
        descriptor = self.fetch_metadata(dataset_id)
        if descriptor is None or not descriptor.files:
            self.logger.warning("No files for %s/%s", self.source_name, dataset_id)
            return None
        # download the first structured file (tab/csv/parquet/xlsx preferred)
        ordered = sorted(
            descriptor.files,
            key=lambda f: {"csv": 0, "tsv": 1, "tab": 2, "parquet": 3, "xlsx": 4, "xls": 5}.get(
                (f.get("format") or "").lower(), 99
            ),
        )
        chosen = ordered[0]
        content = self.http.get_bytes(chosen["url"])
        if content is None:
            return None
        safe_name = (chosen.get("name") or f"{dataset_id.split('/')[-1]}.csv").replace("/", "_")
        dest = target_dir / safe_name
        dest.write_bytes(content)
        return dest

    def to_records(
        self, dataset_id: str, download_path: Path | None = None
    ) -> list[AgriculturalRecord]:
        """Emit a single provenance record describing the downloaded dataset."""
        if download_path is None:
            return []
        import hashlib

        checksum = hashlib.sha256(download_path.read_bytes()).hexdigest()[:16]
        return [
            AgriculturalRecord(
                source=self.source_name,
                dataset_id=dataset_id,
                variable="dataset",
                value=None,
                unit=None,
                provenance=f"{self.instance_url}/dataset.xhtml?persistentId=doi:{dataset_id}",
                license=None,
                extra={
                    "download_path": str(download_path),
                    "file_size_bytes": download_path.stat().st_size,
                    "sha256": checksum,
                },
            )
        ]


class HarvardDataverseConnector(DataverseConnector):
    instance_url = "https://dataverse.harvard.edu"
