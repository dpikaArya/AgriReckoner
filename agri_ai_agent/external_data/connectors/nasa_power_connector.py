import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


class NASAPowerConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "NASA_POWER"

    @property
    def base_url(self) -> str:
        return "https://power.larc.nasa.gov/api"

    def connect(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/temporal/monthly/point", timeout=10)
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def discover(self, query: Optional[str] = None) -> list[dict]:
        return [
            {
                "id": "monthly",
                "name": "Monthly Agroclimatology",
                "description": "Monthly average parameters (temperature, precipitation, solar radiation)",
                "parameters": ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "RH2M", "WS2M"],
            },
            {
                "id": "daily",
                "name": "Daily Agroclimatology",
                "description": "Daily average parameters starting from 1981",
                "parameters": ["T2M_MAX", "T2M_MIN", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
            },
            {
                "id": "climatology",
                "name": "Annual Climatology",
                "description": "Long-term monthly/annual averages",
                "parameters": ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"],
            },
        ]

    def download(self, resource_id: str, target_dir: Path) -> Optional[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        temporal = resource_id
        params = {
            "request": "execute",
            "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M",
            "startDate": "20200101",
            "endDate": "20201231",
            "latitude": "28.5",
            "longitude": "77.0",
            "userCommunity": "AG",
            "format": "CSV",
        }
        if temporal == "daily":
            params["startDate"] = "20200101"
            params["endDate"] = "20201231"
        elif temporal == "climatology":
            params.pop("startDate", None)
            params.pop("endDate", None)

        local_path = target_dir / f"nasa_power_{temporal}.csv"
        try:
            resp = requests.get(
                f"{self.base_url}/temporal/{temporal}/point",
                params=params,
                timeout=120,
            )
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            return local_path
        except requests.RequestException:
            return None

    def validate(self, package: DatasetPackage) -> bool:
        package.validation_errors.clear()
        if package.data is None and package.download_path is None:
            package.validation_errors.append("No data or download path")
            return False
        df = package.data if package.data is not None else package.to_dataframe()
        if df.empty:
            package.validation_errors.append("Empty dataset")
            return False
        package.is_valid = True
        return True

    def register(self, package: DatasetPackage) -> str:
        checksum = hashlib.sha256(str(package.download_path).encode()).hexdigest()[:16]
        package.checksum = checksum
        return checksum

    def update(self) -> int:
        return 0

    def close(self) -> None:
        pass
