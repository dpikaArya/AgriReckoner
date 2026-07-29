from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DatasetPackage:
    source: str
    resource_id: str
    name: str
    description: str = ""
    download_path: Optional[Path] = None
    data: Optional[pd.DataFrame] = None
    metadata: dict = field(default_factory=dict)
    downloaded_at: Optional[datetime] = None
    checksum: Optional[str] = None
    row_count: int = 0
    column_count: int = 0
    validation_errors: list[str] = field(default_factory=list)
    is_valid: bool = False

    def __post_init__(self):
        if self.data is not None:
            self.row_count = len(self.data)
            self.column_count = len(self.data.columns)
        if self.downloaded_at is None:
            self.downloaded_at = datetime.now()

    def to_dataframe(self) -> pd.DataFrame:
        if self.data is not None:
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
            return self.data
        return pd.DataFrame()
