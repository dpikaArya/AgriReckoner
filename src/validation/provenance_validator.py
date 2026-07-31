import hashlib
import json
import logging
from pathlib import Path


class ProvenanceValidator:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def validate_checksum(
        self, file_path: Path, expected_hash: str, algorithm: str = "sha256"
    ) -> dict:
        if not file_path.exists():
            return {
                "valid": False,
                "error": f"File not found: {file_path}",
                "algorithm": algorithm,
                "computed_hash": None,
                "expected_hash": expected_hash,
            }

        try:
            h = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            computed = h.hexdigest()
            match = computed == expected_hash
            return {
                "valid": match,
                "algorithm": algorithm,
                "computed_hash": computed,
                "expected_hash": expected_hash,
                "error": None
                if match
                else f"Hash mismatch: computed={computed}, expected={expected_hash}",
            }
        except ValueError:
            return {
                "valid": False,
                "error": f"Unsupported hash algorithm: {algorithm}",
                "algorithm": algorithm,
                "computed_hash": None,
                "expected_hash": expected_hash,
            }

    def validate_manifest(self, manifest_path: Path) -> dict:
        if not manifest_path.exists():
            return {"valid": False, "error": f"Manifest file not found: {manifest_path}"}

        try:
            with open(manifest_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {"valid": False, "error": f"Invalid JSON in manifest: {e}"}

        required_fields = [
            "dataset_name",
            "source_api",
            "download_timestamp",
            "version",
            "record_count",
            "column_schema",
            "file_hash",
        ]
        missing = [f for f in required_fields if f not in data]
        extra = [k for k in data if k not in required_fields]

        field_errors = []
        if missing:
            field_errors.append(f"Missing required fields: {missing}")
        if not isinstance(data.get("record_count"), (int, float)):
            field_errors.append("record_count must be numeric")

        return {
            "valid": len(field_errors) == 0,
            "manifest_path": str(manifest_path),
            "dataset_name": data.get("dataset_name"),
            "version": data.get("version"),
            "record_count": data.get("record_count"),
            "missing_fields": missing,
            "extra_fields": extra,
            "field_errors": field_errors,
        }

    def validate_metadata(self, metadata: dict) -> dict:
        required_fields = [
            "dataset_name",
            "source",
            "api_endpoint",
            "query",
            "download_timestamp",
            "version",
        ]
        missing = [f for f in required_fields if f not in metadata]
        extra = [k for k in metadata if k not in required_fields]

        return {
            "valid": len(missing) == 0,
            "missing_fields": missing,
            "extra_fields": extra,
            "field_count": len(metadata),
        }

    def validate_lineage(self, lineage: dict) -> dict:
        required_stages = [
            "api_source",
            "raw_dataset",
            "clean_dataset",
            "feature_dataset",
            "training_dataset",
            "model",
        ]
        present_stages = [s for s in required_stages if s in lineage]
        missing_stages = [s for s in required_stages if s not in lineage]
        extra_stages = [k for k in lineage if k not in required_stages]

        return {
            "valid": len(missing_stages) == 0,
            "present_stages": present_stages,
            "missing_stages": missing_stages,
            "extra_stages": extra_stages,
            "total_stages": len(lineage),
        }
