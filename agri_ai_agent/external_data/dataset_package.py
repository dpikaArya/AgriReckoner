import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class DatasetPackage:
    dataset_id: str = ""
    provider: str = ""
    document_type: str = "tabular"
    metadata: dict = field(default_factory=dict)
    documents: list[dict] = field(default_factory=list)
    tables: list[pd.DataFrame] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    supplementary_files: list[Path] = field(default_factory=list)
    schema: dict = field(default_factory=dict)
    statistics: dict = field(default_factory=dict)
    checksum: str = ""
    license: str = ""

    source: str = ""
    resource_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    source_url: str = ""
    download_path: Path | None = None
    data: pd.DataFrame | None = None
    downloaded_at: datetime | None = None
    row_count: int = 0
    column_count: int = 0
    validation_errors: list[str] = field(default_factory=list)
    is_valid: bool = False

    def __post_init__(self):
        if not self.provider and self.source:
            self.provider = self.source
        if self.data is not None:
            self.row_count = len(self.data)
            self.column_count = len(self.data.columns)
            if not self.tables:
                self.tables = [self.data]
        if self.downloaded_at is None:
            self.downloaded_at = datetime.now()
        if not self.dataset_id:
            self.dataset_id = self._generate_id()
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _generate_id(self) -> str:
        raw = f"{self.provider or self.source}/{self.resource_id}/{self.version}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _compute_checksum(self) -> str:
        raw = json.dumps(
            {
                "provider": self.provider or self.source,
                "resource_id": self.resource_id,
                "version": self.version,
                "name": self.name,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dataframe(self) -> pd.DataFrame:
        if self.data is not None:
            return self.data
        if self.tables:
            self.data = self.tables[0]
            return self.data
        if self.download_path and self.download_path.exists():
            suffix = self.download_path.suffix.lower()
            if suffix == ".csv":
                self.data = pd.read_csv(self.download_path)
            elif suffix in (".xlsx", ".xls"):
                self.data = pd.read_excel(self.download_path)
            elif suffix == ".parquet":
                self.data = pd.read_parquet(self.download_path)
            elif suffix == ".json":
                self.data = pd.read_json(self.download_path)
            self.row_count = len(self.data) if self.data is not None else 0
            self.column_count = len(self.data.columns) if self.data is not None else 0
            if self.data is not None and not self.tables:
                self.tables = [self.data]
            return self.data
        return pd.DataFrame()

    def build_schema(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            self.schema = {}
        else:
            self.schema = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "null_counts": {col: int(df[col].isna().sum()) for col in df.columns},
                "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
            }
        return self.schema

    def build_statistics(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            self.statistics = {}
        else:
            numeric = df.select_dtypes(include="number")
            self.statistics = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "numeric_columns": len(numeric.columns),
                "numeric_stats": numeric.describe().to_dict() if not numeric.empty else {},
            }
        return self.statistics

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "provider": self.provider or self.source,
            "document_type": self.document_type,
            "metadata": self.metadata,
            "documents": self.documents,
            "tables": len(self.tables),
            "images": len(self.images),
            "supplementary_files": [str(p) for p in self.supplementary_files],
            "schema": self.schema,
            "statistics": self.statistics,
            "checksum": self.checksum,
            "license": self.license,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_raw(
        cls,
        source: str = "",
        resource_id: str = "",
        name: str = "",
        download_path: Path | None = None,
        data: pd.DataFrame | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> "DatasetPackage":
        return cls(
            provider=source,
            source=source,
            resource_id=resource_id,
            name=name,
            download_path=download_path,
            data=data,
            metadata=metadata or {},
            document_type="tabular"
            if data is not None
            or (
                download_path
                and download_path.suffix.lower() in (".csv", ".xlsx", ".xls", ".parquet", ".json")
            )
            else "unknown",
            tables=[data] if data is not None else [],
            **kwargs,
        )
