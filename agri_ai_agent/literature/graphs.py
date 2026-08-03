"""Graph construction: citation graph, author co-authorship graph and
experiment graph, exported as DataFrame-friendly edge lists."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from agri_ai_agent.literature.connector import normalize_doi
from agri_ai_agent.literature.models import LiteratureRecord


def _paper_key(record: LiteratureRecord) -> str:
    return record.paper_id or record.source_id


def build_citation_graph(records: list[LiteratureRecord]) -> pd.DataFrame:
    """Edges: source paper -> cited paper (resolved by DOI when possible)."""
    rows: list[dict[str, Any]] = []
    by_doi: dict[str, str] = {}
    for record in records:
        doi = normalize_doi(record.doi)
        if doi:
            by_doi.setdefault(doi, _paper_key(record))

    for record in records:
        src = _paper_key(record)
        refs = list(record.references or [])
        refs += [normalize_doi(r) for r in refs if r]
        for ref in refs:
            ref_doi = normalize_doi(ref)
            target = by_doi.get(ref_doi or "") if ref_doi else None
            rows.append(
                {
                    "source_paper_id": src,
                    "source_doi": record.canonical_doi or normalize_doi(record.doi),
                    "target_paper_id": target or "",
                    "target_doi": ref_doi,
                    "source": record.source,
                    "relation": "cites",
                }
            )
    return pd.DataFrame(rows)


def build_author_graph(records: list[LiteratureRecord]) -> pd.DataFrame:
    """Co-authorship edges weighted by number of shared papers."""
    paper_authors: list[tuple[str, list[str]]] = []
    for record in records:
        authors = [a.full_name for a in record.authors if a.full_name]
        if authors:
            paper_authors.append((_paper_key(record), authors))

    edge_counts: Counter[tuple[str, str]] = Counter()
    edge_papers: dict[tuple[str, str], list[str]] = defaultdict(list)
    for paper_id, authors in paper_authors:
        unique = list(dict.fromkeys(authors))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                key = tuple(sorted((unique[i], unique[j])))
                edge_counts[key] += 1
                edge_papers[key].append(paper_id)

    rows = [
        {
            "author_a": a,
            "author_b": b,
            "shared_papers": count,
            "paper_ids": ";".join(edge_papers[(a, b)]),
        }
        for (a, b), count in edge_counts.items()
    ]
    return pd.DataFrame(rows)


def build_experiment_graph(records: list[LiteratureRecord]) -> pd.DataFrame:
    """Experiment graph: papers connected by shared crops, designs or
    variables — the de-facto 'similar experiment' network."""
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            shared_crops = set(a.crop_terms) & set(b.crop_terms)
            shared_designs = set(a.experimental_design) & set(b.experimental_design)
            shared_vars = set(a.study_variables) & set(b.study_variables)
            if shared_crops or shared_designs or shared_vars:
                rows.append(
                    {
                        "source_paper_id": _paper_key(a),
                        "target_paper_id": _paper_key(b),
                        "shared_crops": ";".join(sorted(shared_crops)),
                        "shared_designs": ";".join(sorted(shared_designs)),
                        "shared_variables": ";".join(sorted(shared_vars)),
                        "similarity": round(
                            0.4 * len(shared_crops)
                            + 0.3 * len(shared_designs)
                            + 0.3 * len(shared_vars),
                            3,
                        ),
                    }
                )
    return pd.DataFrame(rows)
