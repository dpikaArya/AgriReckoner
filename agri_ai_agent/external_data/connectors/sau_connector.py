import hashlib
from pathlib import Path
from typing import Optional

import requests

from agri_ai_agent.external_data.connector import ExternalDataConnector
from agri_ai_agent.external_data.dataset_package import DatasetPackage


_SAU_REGISTRY = [
    {
        "id": "pau",
        "name": "Punjab Agricultural University",
        "url": "https://www.pau.edu",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=pau",
    },
    {
        "id": "gbpuat",
        "name": "G.B. Pant University of Agriculture and Technology",
        "url": "https://www.gbpuat.ac.in",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=gbpuat",
    },
    {
        "id": "tanuvas",
        "name": "Tamil Nadu Veterinary and Animal Sciences University",
        "url": "https://www.tanuvas.ac.in",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=tanuvas",
    },
    {
        "id": "angrau",
        "name": "Acharya N.G. Ranga Agricultural University",
        "url": "https://www.angrau.ac.in",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=angrau",
    },
    {
        "id": "uasb",
        "name": "University of Agricultural Sciences, Bangalore",
        "url": "https://www.uasbangalore.edu.in",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=uas+bangalore",
    },
    {
        "id": "uasi",
        "name": "University of Agricultural Sciences, Dharwad",
        "url": "https://www.uasd.edu",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=uas+dharwad",
    },
    {
        "id": "mpuat",
        "name": "Maharana Pratap University of Agriculture and Technology",
        "url": "https://www.mpuat.ac.in",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=mpuat",
    },
    {
        "id": "sknau",
        "name": "Sri Karan Narendra Agriculture University",
        "url": "https://www.sknau.ac.in",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=sknau",
    },
    {
        "id": "jau",
        "name": "Junagadh Agricultural University",
        "url": "https://www.jau.in",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=junagadh",
    },
    {
        "id": "nduat",
        "name": "Narendra Deva University of Agriculture and Technology",
        "url": "https://www.nduat.org",
        "api": "https://krishikosh.egranth.ac.in/api/search?q=nduat",
    },
]


class SAUConnector(ExternalDataConnector):
    @property
    def source_name(self) -> str:
        return "State_Agricultural_Universities"

    @property
    def base_url(self) -> str:
        return "https://krishikosh.egranth.ac.in"

    def connect(self) -> bool:
        try:
            resp = requests.head(self.base_url, timeout=10)
            return resp.status_code < 500
        except requests.RequestException:
            return False

    def discover(self, query: Optional[str] = None) -> list[dict]:
        if query:
            q = query.lower()
            return [sau for sau in _SAU_REGISTRY if q in sau["name"].lower() or q in sau["id"]]
        return list(_SAU_REGISTRY)

    def download(self, resource_id: str, target_dir: Path) -> Optional[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        matching = [s for s in _SAU_REGISTRY if s["id"] == resource_id]
        if not matching:
            return None
        sau = matching[0]
        try:
            resp = requests.get(
                sau["api"],
                params={"limit": 50},
                timeout=60,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "html" in content_type.lower():
                return None

            if "json" in content_type:
                import pandas as pd
                data = resp.json()
                records = data if isinstance(data, list) else data.get("records", data.get("results", [data]))
                if isinstance(records, list) and len(records) > 0:
                    df = pd.DataFrame(records)
                    local_path = target_dir / f"sau_{resource_id}.parquet"
                    df.to_parquet(local_path, index=False)
                    return local_path

            return None
        except requests.RequestException:
            return None

    def validate(self, package: DatasetPackage) -> bool:
        package.validation_errors.clear()
        if package.download_path is None:
            package.validation_errors.append("No download path")
            return False
        if not package.download_path.exists() or package.download_path.stat().st_size == 0:
            package.validation_errors.append("File missing or empty")
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
