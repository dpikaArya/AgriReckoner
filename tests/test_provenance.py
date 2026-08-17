import json
from pathlib import Path

import pytest

from src.provenance.lineage_tracker import (
    LineageTracker,
    compute_checksum,
    generate_manifest,
    generate_metadata,
)


class TestProvenanceMetadata:
    def test_generate_metadata_required_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        metadata = generate_metadata(
            dataset_name="test_ds",
            source_api="openweather",
            api_endpoint="/data/2.5/weather",
            query_params={"q": "London"},
            version="1.0",
            record_count=100,
            column_schema={"temp": "float"},
        )
        assert metadata["dataset_name"] == "test_ds"
        assert metadata["source_api"] == "openweather"
        assert metadata["api_endpoint"] == "/data/2.5/weather"
        assert metadata["query_params"] == {"q": "London"}
        assert metadata["version"] == "1.0"
        assert metadata["record_count"] == 100
        assert "download_timestamp" in metadata

    def test_generate_metadata_raises_on_missing_name(self):
        with pytest.raises(ValueError, match="dataset_name is required"):
            generate_metadata("", "source", "/ep", {}, "1.0", 10, {})

    def test_generate_metadata_raises_on_negative_records(self):
        with pytest.raises(ValueError, match="record_count must be non-negative"):
            generate_metadata("ds", "src", "/ep", {}, "1.0", -1, {"a": "int"})


class TestProvenanceManifest:
    def test_generate_manifest_structure(self, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        manifest = generate_manifest(
            dataset_name="test_ds",
            source="openweather",
            file_paths=[data_file],
            record_count=1,
            column_schema={"a": "int", "b": "int"},
        )
        assert manifest["dataset_name"] == "test_ds"
        assert manifest["source"] == "openweather"
        assert "file_hashes" in manifest
        assert "generated_at" in manifest
        assert str(data_file) in manifest["file_hashes"]

    def test_generate_manifest_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_manifest("ds", "src", [tmp_path / "nonexistent.csv"], 0, {"a": "int"})

    def test_generate_manifest_raises_on_negative_count(self, tmp_path):
        with pytest.raises(ValueError, match="record_count must be non-negative"):
            generate_manifest("ds", "src", [], -1, {"a": "int"})


class TestComputeChecksum:
    def test_compute_checksum_sha256(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        digest = compute_checksum(f)
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_compute_checksum_raises_on_missing(self):
        with pytest.raises(FileNotFoundError):
            compute_checksum(Path("/nonexistent/file.csv"))


class TestLineageTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        lineage_dir = tmp_path / "lineage"
        return LineageTracker(lineage_dir=lineage_dir)

    def test_initialize_returns_non_empty_id(self, tracker):
        dataset_id = tracker.initialize_dataset("test_ds", {"source": "test"})
        assert isinstance(dataset_id, str)
        assert len(dataset_id) > 0

    def test_initialize_creates_graph_file(self, tracker):
        tracker.initialize_dataset("test_ds", {"source": "test"})
        graph_file = tracker.lineage_dir / "lineage_graph.json"
        assert graph_file.exists()
        data = json.loads(graph_file.read_text())
        assert len(data) == 1
        assert data[0]["dataset_name"] == "test_ds"

    def test_transition_records_stage_change(self, tracker):
        dataset_id = tracker.initialize_dataset("test_ds", {"source": "test"})
        artifact = tracker.lineage_dir / "raw.csv"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("raw data")
        tracker.transition(dataset_id, "api_source", "raw_dataset", artifact)
        graph = json.loads((tracker.lineage_dir / "lineage_graph.json").read_text())
        transitions = [r for r in graph if r["to_stage"] == "raw_dataset"]
        assert len(transitions) == 1

    def test_get_lineage_returns_ordered_stages(self, tracker):
        dataset_id = tracker.initialize_dataset("test_ds", {"source": "test"})
        artifact = tracker.lineage_dir / "raw.csv"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("raw")
        tracker.transition(dataset_id, "api_source", "raw_dataset", artifact)
        lineage = tracker.get_lineage(dataset_id)
        assert len(lineage) == 2
        assert lineage[0]["to_stage"] == "api_source"
        assert lineage[1]["to_stage"] == "raw_dataset"

    def test_invalid_transition_raises(self, tracker):
        dataset_id = tracker.initialize_dataset("test_ds", {"source": "test"})
        artifact = tracker.lineage_dir / "dummy.txt"
        artifact.write_text("x")
        with pytest.raises(ValueError, match="Invalid transition"):
            tracker.transition(dataset_id, "api_source", "training_dataset", artifact)

    def test_get_lineage_raises_on_unknown_id(self, tracker):
        with pytest.raises(ValueError, match="No lineage records found"):
            tracker.get_lineage("nonexistent-id")

    def test_initialize_raises_on_empty_name(self, tracker):
        with pytest.raises(ValueError, match="dataset_name is required"):
            tracker.initialize_dataset("", {})
