import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

STAGES = [
    "api_source",
    "raw_dataset",
    "clean_dataset",
    "feature_dataset",
    "training_dataset",
    "model",
]


@dataclass
class DatasetProvenance:
    dataset_name: str
    source_api: str
    api_endpoint: str
    query_params: dict
    download_timestamp: str
    version: str
    doi: str | None = None
    license: str = ""
    authors: list | None = None
    citation: str = ""
    file_hash: str = ""
    record_count: int = 0
    column_schema: dict | None = None
    size_bytes: int = 0

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.column_schema is None:
            self.column_schema = {}


def compute_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    h = hashlib.new(algorithm)
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    checksum = h.hexdigest()

    checksum_file = file_path.with_name(f"checksum.{algorithm}")
    checksum_file.write_text(f"{checksum}  {file_path.name}\n")
    return checksum


def generate_metadata(
    dataset_name: str,
    source_api: str,
    api_endpoint: str,
    query_params: dict,
    version: str,
    record_count: int,
    column_schema: dict,
    doi: str = "",
    license: str = "",
    authors: list | None = None,
    citation: str = "",
) -> dict:
    if not dataset_name:
        raise ValueError("dataset_name is required")
    if not source_api:
        raise ValueError("source_api is required")
    if not api_endpoint:
        raise ValueError("api_endpoint is required")
    if not version:
        raise ValueError("version is required")
    if record_count < 0:
        raise ValueError("record_count must be non-negative")
    if not column_schema:
        raise ValueError("column_schema is required")

    provenance = DatasetProvenance(
        dataset_name=dataset_name,
        source_api=source_api,
        api_endpoint=api_endpoint,
        query_params=query_params,
        download_timestamp=datetime.now(timezone.utc).isoformat(),
        version=version,
        doi=doi,
        license=license,
        authors=authors or [],
        citation=citation,
        record_count=record_count,
        column_schema=column_schema,
    )

    metadata = asdict(provenance)

    lineage_dir = Path("data") / "lineage" / dataset_name
    lineage_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = lineage_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    return metadata


def generate_manifest(
    dataset_name: str,
    source: str,
    file_paths: list[Path],
    record_count: int,
    column_schema: dict,
) -> dict:
    if not dataset_name:
        raise ValueError("dataset_name is required")
    if not source:
        raise ValueError("source is required")
    if record_count < 0:
        raise ValueError("record_count must be non-negative")
    if not column_schema:
        raise ValueError("column_schema is required")

    file_hashes = {}
    for fp in file_paths:
        fp = Path(fp)
        if not fp.is_file():
            raise FileNotFoundError(f"File not found: {fp}")
        file_hashes[str(fp)] = compute_checksum(fp)

    manifest = {
        "dataset_name": dataset_name,
        "source": source,
        "file_hashes": file_hashes,
        "record_count": record_count,
        "column_schema": column_schema,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    lineage_dir = Path("data") / "lineage" / dataset_name
    lineage_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = lineage_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    return manifest


def _load_graph(lineage_dir: Path) -> list[dict]:
    graph_file = lineage_dir / "lineage_graph.json"
    if graph_file.is_file():
        return json.loads(graph_file.read_text())
    return []


def _save_graph(lineage_dir: Path, graph: list[dict]) -> None:
    graph_file = lineage_dir / "lineage_graph.json"
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    graph_file.write_text(json.dumps(graph, indent=2, default=str))


class LineageTracker:
    def __init__(self, lineage_dir: Path = Path("data/lineage")):
        self.lineage_dir = Path(lineage_dir).resolve()
        self.lineage_dir.mkdir(parents=True, exist_ok=True)

    def initialize_dataset(self, dataset_name: str, metadata: dict) -> str:
        if not dataset_name:
            raise ValueError("dataset_name is required")
        if not metadata:
            raise ValueError("metadata is required")

        dataset_id = str(uuid.uuid4())

        init_record = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "from_stage": None,
            "to_stage": "api_source",
            "artifact_path": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
        }

        graph = _load_graph(self.lineage_dir)
        graph.append(init_record)
        _save_graph(self.lineage_dir, graph)

        return dataset_id

    def transition(
        self,
        dataset_id: str,
        from_stage: str,
        to_stage: str,
        artifact_path: Path,
        metadata: dict | None = None,
    ) -> None:
        if not dataset_id:
            raise ValueError("dataset_id is required")
        if from_stage not in STAGES:
            raise ValueError(f"Invalid from_stage '{from_stage}'. Must be one of {STAGES}")
        if to_stage not in STAGES:
            raise ValueError(f"Invalid to_stage '{to_stage}'. Must be one of {STAGES}")

        from_index = STAGES.index(from_stage)
        to_index = STAGES.index(to_stage)
        if to_index != from_index + 1:
            raise ValueError(
                f"Invalid transition: '{from_stage}' -> '{to_stage}'. "
                f"Only sequential transitions are allowed."
            )

        graph = _load_graph(self.lineage_dir)

        dataset_name = None
        for record in graph:
            if record["dataset_id"] == dataset_id:
                dataset_name = record.get("dataset_name")
                break

        if dataset_name is None:
            raise ValueError(
                f"Dataset with id '{dataset_id}' not found. Call initialize_dataset first."
            )

        transition_record = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "artifact_path": str(Path(artifact_path).resolve()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        graph.append(transition_record)
        _save_graph(self.lineage_dir, graph)

    def get_lineage(self, dataset_id: str) -> list[dict]:
        if not dataset_id:
            raise ValueError("dataset_id is required")

        graph = _load_graph(self.lineage_dir)
        lineage = [record for record in graph if record["dataset_id"] == dataset_id]
        if not lineage:
            raise ValueError(f"No lineage records found for dataset id '{dataset_id}'")
        return sorted(lineage, key=lambda r: r["timestamp"])

    def get_upstream(self, model_name: str) -> list[dict]:
        if not model_name:
            raise ValueError("model_name is required")

        graph = _load_graph(self.lineage_dir)

        model_records = [
            r for r in graph if r.get("dataset_name") == model_name and r["to_stage"] == "model"
        ]
        if not model_records:
            raise ValueError(f"No model records found for '{model_name}'")

        dataset_ids = set()
        for r in model_records:
            dataset_ids.add(r["dataset_id"])

        visited: set[str] = set()
        queue: list[str] = list(dataset_ids)
        upstream: list[dict] = []

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            records = [r for r in graph if r["dataset_id"] == current_id]
            upstream.extend(records)

            for r in records:
                parent_id = r.get("parent_dataset_id")
                if parent_id and parent_id not in visited:
                    queue.append(parent_id)

        return sorted(upstream, key=lambda r: r["timestamp"])

    def get_downstream(self, dataset_id: str) -> list[str]:
        if not dataset_id:
            raise ValueError("dataset_id is required")

        graph = _load_graph(self.lineage_dir)

        dataset_records = [r for r in graph if r["dataset_id"] == dataset_id]
        if not dataset_records:
            raise ValueError(f"No records found for dataset id '{dataset_id}'")

        dataset_name = dataset_records[0].get("dataset_name")

        downstream: set[str] = set()
        for r in graph:
            parent_id = r.get("parent_dataset_id")
            if parent_id == dataset_id:
                downstream.add(r.get("dataset_name", ""))

        name_downstream: set[str] = set()
        for r in graph:
            if r.get("parent_dataset_name") == dataset_name:
                name_downstream.add(r.get("dataset_name", ""))

        return sorted(downstream | name_downstream)

    def visualize(self, output_path: Path | None = None) -> str:
        graph = _load_graph(self.lineage_dir)
        if not graph:
            return "digraph Lineage {}"

        nodes: dict[str, set[str]] = {}
        for record in graph:
            did = record["dataset_id"]
            if did not in nodes:
                nodes[did] = set()
            nodes[did].add(record["to_stage"])
            if record["from_stage"]:
                nodes[did].add(record["from_stage"])

        lines = ["digraph Lineage {"]
        lines.append("    rankdir=LR;")
        lines.append('    node [shape=box, style="rounded,filled", fillcolor=lightyellow];')
        lines.append("")

        for did, stages in nodes.items():
            dname = graph[[r["dataset_id"] == did for r in graph].index(True)]["dataset_name"]
            label = f"{dname}\\n{' | '.join(sorted(stages))}"
            node_id = did.replace("-", "_")
            lines.append(f'    {node_id} [label="{label}"];')

        edges: set[tuple[str, str]] = set()
        for record in graph:
            if record["from_stage"] and record["to_stage"]:
                src = record["from_stage"].replace("_", " ")
                dst = record["to_stage"].replace("_", " ")
                edges.add((src, dst))

        lines.append("")
        for src, dst in sorted(edges):
            lines.append(f'    "{src}" -> "{dst}";')

        lines.append("}")
        dot_str = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(dot_str)

        return dot_str
