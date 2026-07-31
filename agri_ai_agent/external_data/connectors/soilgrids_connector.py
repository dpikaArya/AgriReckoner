import hashlib
import os
from pathlib import Path

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


def _get_sg_config(key: str, default: str) -> str:
    env_key = f"AGRI_SOILGRIDS_{key.upper()}"
    return os.environ.get(env_key, default)


class SoilGridsConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "SoilGrids"

    @property
    def base_url(self) -> str:
        return "https://rest.isric.org/soilgrids/v2.0"

    def connect(self) -> bool:
        try:
            resp = requests.get(
                f"{self.base_url}/properties/query?lon=0&lat=0&property=clay&depth=0-5cm",
                timeout=15,
            )
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def discover(self, query: str | None = None) -> list[dict]:
        properties = [
            {"id": "bdod", "name": "Bulk density", "unit": "kg/dm³"},
            {"id": "cec", "name": "Cation exchange capacity", "unit": "mmol(c)/kg"},
            {"id": "cfvo", "name": "Coarse fragments", "unit": "cm³/dm³"},
            {"id": "clay", "name": "Clay content", "unit": "g/kg"},
            {"id": "nitrogen", "name": "Nitrogen", "unit": "g/kg"},
            {"id": "phh2o", "name": "Soil pH in H2O", "unit": "pH"},
            {"id": "sand", "name": "Sand content", "unit": "g/kg"},
            {"id": "silt", "name": "Silt content", "unit": "g/kg"},
            {"id": "soc", "name": "Soil organic carbon", "unit": "g/kg"},
            {"id": "ocd", "name": "Organic carbon density", "unit": "kg/m³"},
        ]
        if query:
            q = query.lower()
            properties = [p for p in properties if q in p["name"].lower() or q in p["id"]]
        return [
            {
                "id": p["id"],
                "name": p["name"],
                "description": f"{p['name']} ({p['unit']}) at 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm depth",
                "unit": p["unit"],
                "depths": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"],
            }
            for p in properties
        ]

    def download(self, resource_id: str, target_dir: Path) -> Path | None:
        target_dir.mkdir(parents=True, exist_ok=True)
        lat = _get_sg_config("latitude", "13.0")
        lon = _get_sg_config("longitude", "77.5")
        depth = _get_sg_config("depth", "0-5cm")

        local_path = target_dir / f"soilgrids_{resource_id}_{lat}_{lon}.csv"
        try:
            url = f"{self.base_url}/properties/query"
            params = {
                "lon": lon,
                "lat": lat,
                "property": resource_id,
                "depth": depth,
                "value": "mean",
            }
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            rows = []
            for layer in data.get("properties", {}).get("layers", []):
                prop_name = layer.get("name", resource_id)
                unit = layer.get("unit_measure", {}).get("target_units", "")
                for d_layer in layer.get("depths", []):
                    vals = d_layer.get("values", {})
                    for stat_name in ("mean", "uncertainty"):
                        stat_val = vals.get(stat_name)
                        if stat_val is not None:
                            rows.append(
                                {
                                    "Latitude": float(lat),
                                    "Longitude": float(lon),
                                    "Property": prop_name,
                                    "Depth": d_layer.get("label", depth),
                                    "Statistic": stat_name,
                                    "Value": stat_val,
                                    "Unit": unit,
                                }
                            )
            if not rows:
                rows.append(
                    {
                        "Latitude": float(lat),
                        "Longitude": float(lon),
                        "Property": resource_id,
                        "Depth": depth,
                        "Statistic": "mean",
                        "Value": None,
                        "Unit": "",
                    }
                )
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_csv(local_path, index=False)
            return local_path
        except requests.RequestException:
            return None

    def validate(self, package: DatasetPackage) -> bool:
        package.validation_errors.clear()
        if package.download_path is None:
            package.validation_errors.append("No download path")
            return False
        if not package.download_path.exists():
            package.validation_errors.append("Download path does not exist")
            return False
        if package.download_path.stat().st_size == 0:
            package.validation_errors.append("Empty file")
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
