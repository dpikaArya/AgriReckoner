import json
from pathlib import Path
from typing import Optional

from agri_ai_agent.external_data.dataset_package import DatasetPackage
from agri_ai_agent.knowledge_graph.graph import KnowledgeGraph


def write_provenance_artifact(
    provenance_dir: Path,
    pkg: DatasetPackage,
    endpoint_url: str = "",
    connector_name: str = "",
) -> Path:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset_id": pkg.dataset_id,
        "provider": pkg.provider or pkg.source,
        "resource_id": pkg.resource_id,
        "name": pkg.name,
        "version": pkg.version,
        "connector_name": connector_name or pkg.source,
        "endpoint_url": endpoint_url or pkg.source_url,
        "download_path": str(pkg.download_path) if pkg.download_path else "",
        "checksum": pkg.checksum,
        "row_count": pkg.row_count,
        "column_count": pkg.column_count,
        "is_valid": pkg.is_valid,
        "schema": pkg.schema,
        "statistics": pkg.statistics,
    }
    path = provenance_dir / f"provenance_{pkg.dataset_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def register_dataset_lineage(
    pkg: DatasetPackage,
    endpoint_url: str = "",
    connector_name: str = "",
) -> dict:
    return {
        "dataset_id": pkg.dataset_id,
        "source": pkg.source,
        "provider": pkg.provider or pkg.source,
        "resource_id": pkg.resource_id,
        "version": pkg.version,
        "endpoint_url": endpoint_url or pkg.source_url,
        "connector_name": connector_name or pkg.source,
        "checksum": pkg.checksum,
        "downloaded_at": str(pkg.downloaded_at),
        "row_count": pkg.row_count,
        "column_count": pkg.column_count,
        "tables": len(pkg.tables),
        "documents": len(pkg.documents),
        "images": len(pkg.images),
    }


def register_document_lineage(
    pkg: DatasetPackage,
    document_index: int = 0,
) -> dict:
    return {
        "document_id": f"{pkg.dataset_id}/doc/{document_index}",
        "dataset_id": pkg.dataset_id,
        "source": pkg.source,
        "provider": pkg.provider or pkg.source,
        "document_type": pkg.document_type,
        "resource_id": pkg.resource_id,
        "version": pkg.version,
    }


def update_knowledge_graph(
    kg: KnowledgeGraph,
    pkg: DatasetPackage,
    endpoint_url: str = "",
    connector_name: str = "",
):
    repo_id = f"Repository:{pkg.source or pkg.provider}"
    if kg.get_node(repo_id) is None:
        kg.add_node(
            repo_id, "Repository",
            name=pkg.source or pkg.provider,
            provider=pkg.provider or pkg.source,
            endpoint_url=endpoint_url or pkg.source_url,
            connector=connector_name or pkg.source,
        )

    dataset_id = f"Dataset:{pkg.dataset_id}"
    if kg.get_node(dataset_id) is None:
        kg.add_node(
            dataset_id, "Dataset",
            dataset_id=pkg.dataset_id,
            name=pkg.name,
            resource_id=pkg.resource_id,
            version=pkg.version,
            provider=pkg.provider or pkg.source,
            checksum=pkg.checksum,
            row_count=pkg.row_count,
            column_count=pkg.column_count,
            is_valid=pkg.is_valid,
        )

    if not kg.graph.has_edge(repo_id, dataset_id):
        kg.add_edge(repo_id, dataset_id, "PROVIDES")

    doc_id = f"Document:{pkg.dataset_id}/0"
    if kg.get_node(doc_id) is None:
        kg.add_node(
            doc_id, "Document",
            dataset_id=pkg.dataset_id,
            document_type=pkg.document_type,
            name=pkg.name,
            source=pkg.source,
            version=pkg.version,
        )

    if not kg.graph.has_edge(dataset_id, doc_id):
        kg.add_edge(dataset_id, doc_id, "CONTAINS")

    if not kg.graph.has_edge(doc_id, repo_id):
        kg.add_edge(doc_id, repo_id, "FROM_SOURCE")

    if pkg.statistics:
        evidence_id = f"Evidence:{pkg.dataset_id}/stats"
        if kg.get_node(evidence_id) is None:
            kg.add_node(
                evidence_id, "Evidence",
                dataset_id=pkg.dataset_id,
                row_count=pkg.row_count,
                column_count=pkg.column_count,
                checksum=pkg.checksum,
            )
        if not kg.graph.has_edge(doc_id, evidence_id):
            kg.add_edge(doc_id, evidence_id, "SUPPORTED_BY")
