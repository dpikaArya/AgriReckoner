import hashlib
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


DEFAULT_PARAMS = {
    "latitude": "29.9136",
    "longitude": "77.9975",
    "start_date": "2023",
    "end_date": "2023",
    "parameters": "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M",
}


def _get_config(key: str, default: str) -> str:
    env_key = f"AGRI_NASA_POWER_{key.upper()}"
    return os.environ.get(env_key, default)


class NASAPowerConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "NASA_POWER"

    @property
    def base_url(self) -> str:
        return "https://power.larc.nasa.gov/api"

    def connect(self) -> bool:
        try:
            resp = requests.get(
                f"{self.base_url}/temporal/monthly/point",
                params={"parameters": "T2M", "latitude": "0", "longitude": "0",
                         "start": "2020", "end": "2020", "community": "AG", "format": "JSON"},
                timeout=15,
            )
            return resp.status_code == 200
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

        lat = _get_config("latitude", DEFAULT_PARAMS["latitude"])
        lon = _get_config("longitude", DEFAULT_PARAMS["longitude"])
        params_str = _get_config("parameters", DEFAULT_PARAMS["parameters"])

        start_val = _get_config("start_date", DEFAULT_PARAMS["start_date"])
        end_val = _get_config("end_date", DEFAULT_PARAMS["end_date"])

        # POWER API expects YYYY for monthly/climatology, YYYYMMDD for daily
        if temporal == "climatology":
            start_val = start_val[:4]
            end_val = end_val[:4]

        params = {
            "parameters": params_str,
            "latitude": lat,
            "longitude": lon,
            "start": start_val,
            "end": end_val,
            "community": "AG",
            "format": "CSV",
        }

        local_path = target_dir / f"nasa_power_{temporal}_{lat}_{lon}.csv"
        try:
            resp = requests.get(
                f"{self.base_url}/temporal/{temporal}/point",
                params=params,
                timeout=120,
            )
            resp.raise_for_status()
            raw_text = resp.text

            import io
            import re
            import pandas as pd

            lines = raw_text.split('\n')
            data_start = 0
            header_line = None
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('YEAR') or stripped.startswith('PARAMETER'):
                    data_start = i
                    header_line = stripped
                    break
            if header_line is None:
                return None

            csv_content = header_line + '\n' + '\n'.join(lines[data_start + 1:])
            df = pd.read_csv(io.StringIO(csv_content))

            month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            id_vars = [c for c in df.columns if c not in month_cols and c != 'ANN']

            if all(c in df.columns for c in month_cols) and 'PARAMETER' in df.columns:
                df_long = df.melt(
                    id_vars=id_vars,
                    value_vars=month_cols,
                    var_name='Month',
                    value_name='Value'
                )
                df_long.insert(0, 'Latitude', float(lat))
                df_long.insert(1, 'Longitude', float(lon))
                df_long.to_csv(local_path, index=False)
            else:
                df.insert(0, 'Latitude', float(lat))
                df.insert(1, 'Longitude', float(lon))
                # For daily format YEAR,DOY,PARAM1,PARAM2,..., reshape to long
                if 'DOY' in df.columns and 'YEAR' in df.columns:
                    param_cols = [c for c in df.columns if c not in ('Latitude', 'Longitude', 'YEAR', 'DOY')]
                    df_long = df.melt(
                        id_vars=['Latitude', 'Longitude', 'YEAR', 'DOY'],
                        value_vars=param_cols,
                        var_name='PARAMETER',
                        value_name='Value'
                    )
                    df_long.to_csv(local_path, index=False)
                else:
                    df.to_csv(local_path, index=False)
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
