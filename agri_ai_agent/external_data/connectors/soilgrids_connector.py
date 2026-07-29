import hashlib
from pathlib import Path
from typing import Optional

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


class SoilGridsConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "SoilGrids"

    @property
    def base_url(self) -> str:
        return "https://rest.isric.org/soilgrids/v2.0"

    def connect(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/properties", timeout=10)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def discover(self, query: Optional[str] = None) -> list[dict]:
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

    def download(self, resource_id: str, target_dir: Path) -> Optional[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / f"soilgrids_{resource_id}.tif"
        try:
            url = (
                f"{self.base_url}/properties/{resource_id}/wcs?"
                f"service=WCS&version=2.0.1&request=GetCoverage"
                f"&coverageId={resource_id}_0-5cm_mean"
                f"&format=image/tiff"
            )
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
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
