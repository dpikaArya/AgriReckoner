"""
KnowledgeGraph — structured graph for agricultural research entities.

Node types: Paper, Treatment, Crop, Observation, Yield, Soil, Climate, Management,
           Repository, Dataset, Evidence, Document
Edge types: DESCRIBES, TREATS, GROWS, OBSERVES, YIELDS, HAS_SOIL, HAS_CLIMATE, MANAGES,
           PROVIDES, CONTAINS, SUPPORTED_BY, FROM_SOURCE
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


VALID_NODE_TYPES = frozenset([
    "Paper", "Treatment", "Crop", "Observation",
    "Yield", "Soil", "Climate", "Management",
    "Repository", "Dataset", "Evidence", "Document",
])

VALID_EDGE_TYPES = frozenset([
    "DESCRIBES", "TREATS", "GROWS", "OBSERVES",
    "YIELDS", "HAS_SOIL", "HAS_CLIMATE", "MANAGES",
    "PROVIDES", "CONTAINS", "SUPPORTED_BY", "FROM_SOURCE",
])

EDGE_NODE_PAIRS = {
    "DESCRIBES": ("Paper", "Treatment"),
    "TREATS": ("Paper", "Crop"),
    "GROWS": ("Crop", "Treatment"),
    "OBSERVES": ("Paper", "Observation"),
    "YIELDS": ("Observation", "Yield"),
    "HAS_SOIL": ("Paper", "Soil"),
    "HAS_CLIMATE": ("Paper", "Climate"),
    "MANAGES": ("Treatment", "Management"),
    "PROVIDES": ("Repository", "Dataset"),
    "CONTAINS": ("Dataset", "Document"),
    "SUPPORTED_BY": ("Document", "Evidence"),
    "FROM_SOURCE": ("Document", "Repository"),
}


class KnowledgeGraph:
    def __init__(self):
        self._graph = nx.DiGraph()
        self._node_types: dict[str, str] = {}
        self._edge_types: dict[tuple[str, str], str] = {}

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def add_node(self, node_id: str, node_type: str, **properties):
        if node_type not in VALID_NODE_TYPES:
            raise ValueError(f"Invalid node type '{node_type}'. Must be one of {sorted(VALID_NODE_TYPES)}")
        self._graph.add_node(node_id, node_type=node_type, **properties)
        self._node_types[node_id] = node_type

    def add_edge(self, source: str, target: str, edge_type: str, **properties):
        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(f"Invalid edge type '{edge_type}'. Must be one of {sorted(VALID_EDGE_TYPES)}")
        expected = EDGE_NODE_PAIRS[edge_type]
        src_type = self._node_types.get(source)
        tgt_type = self._node_types.get(target)
        if src_type and src_type != expected[0]:
            raise ValueError(
                f"Edge '{edge_type}' requires source of type '{expected[0]}', got '{src_type}'"
            )
        if tgt_type and tgt_type != expected[1]:
            raise ValueError(
                f"Edge '{edge_type}' requires target of type '{expected[1]}', got '{tgt_type}'"
            )
        self._graph.add_edge(source, target, edge_type=edge_type, **properties)
        self._edge_types[(source, target)] = edge_type

    def get_node(self, node_id: str) -> Optional[dict]:
        if node_id in self._graph:
            data = dict(self._graph.nodes[node_id])
            data["id"] = node_id
            return data
        return None

    def get_nodes_by_type(self, node_type: str) -> list[dict]:
        results = []
        for nid, data in self._graph.nodes(data=True):
            if data.get("node_type") == node_type:
                entry = dict(data)
                entry["id"] = nid
                results.append(entry)
        return results

    def get_edges_by_type(self, edge_type: str) -> list[dict]:
        results = []
        for u, v, data in self._graph.edges(data=True):
            if data.get("edge_type") == edge_type:
                entry = dict(data)
                entry["source"] = u
                entry["target"] = v
                results.append(entry)
        return results

    def get_neighbors(self, node_id: str, direction: str = "both") -> list[str]:
        if node_id not in self._graph:
            return []
        if direction == "out":
            return list(self._graph.successors(node_id))
        elif direction == "in":
            return list(self._graph.predecessors(node_id))
        return list(self._graph.predecessors(node_id)) + list(self._graph.successors(node_id))

    def shortest_path(self, source: str, target: str) -> Optional[list[str]]:
        try:
            return nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def centrality(self, method: str = "degree") -> dict[str, float]:
        if method == "degree":
            return nx.degree_centrality(self._graph)
        elif method == "betweenness":
            return nx.betweenness_centrality(self._graph)
        elif method == "in_degree":
            return nx.in_degree_centrality(self._graph)
        elif method == "out_degree":
            return nx.out_degree_centrality(self._graph)
        raise ValueError(f"Unknown centrality method '{method}'")

    def summary(self) -> dict:
        type_counts = Counter(self._node_types.values())
        edge_counts = Counter(self._edge_types.values())
        isolated = list(nx.isolates(self._graph))
        components = nx.weakly_connected_components(self._graph)
        component_sizes = [len(c) for c in components]

        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "node_type_counts": dict(type_counts),
            "edge_type_counts": dict(edge_counts),
            "isolated_nodes": len(isolated),
            "num_components": len(component_sizes),
            "largest_component_size": max(component_sizes) if component_sizes else 0,
            "density": nx.density(self._graph),
        }

    def build_from_dataframe(self, df: pd.DataFrame, paper_col: str = "Source_Paper",
                              crop_col: str = "Crop", treatment_cols: list[str] | None = None):
        if treatment_cols is None:
            treatment_cols = ["Fertilizer_Name", "Dose", "Application_Interval"]

        papers_added = set()
        crops_added = set()
        treatments_added = set()
        edges_added = 0

        for _, row in df.iterrows():
            paper = str(row.get(paper_col, "")).strip()
            crop = str(row.get(crop_col, "")).strip()

            if not paper or paper == "nan":
                continue
            if paper not in papers_added:
                year = row.get("Year", "")
                country = row.get("Country", "")
                doi = row.get("DOI", "")
                self.add_node(
                    f"Paper:{paper}", "Paper",
                    title=paper, year=year, country=country, doi=doi,
                )
                papers_added.add(paper)

            if crop and crop != "nan" and crop not in crops_added:
                self.add_node(f"Crop:{crop}", "Crop", name=crop)
                crops_added.add(crop)

            if paper and crop and crop != "nan":
                if not self._graph.has_edge(f"Paper:{paper}", f"Crop:{crop}"):
                    self.add_edge(f"Paper:{paper}", f"Crop:{crop}", "TREATS")
                    edges_added += 1

            for tcol in treatment_cols:
                tval = str(row.get(tcol, "")).strip()
                if tval and tval != "nan":
                    tid = f"Treatment:{paper}:{tcol}={tval}"
                    if tid not in treatments_added:
                        self.add_node(tid, "Treatment", column=tcol, value=tval, paper=paper)
                        treatments_added.add(tid)
                    if not self._graph.has_edge(f"Paper:{paper}", tid):
                        self.add_edge(f"Paper:{paper}", tid, "DESCRIBES")
                        edges_added += 1
                    if crop and crop != "nan" and not self._graph.has_edge(f"Crop:{crop}", tid):
                        self.add_edge(f"Crop:{crop}", tid, "GROWS")
                        edges_added += 1

            ph = row.get("Soil_pH")
            oc = row.get("Organic_Carbon")
            if pd.notna(ph) or pd.notna(oc):
                soil_id = f"Soil:{paper}"
                if soil_id not in [n for n in self._graph.nodes() if n.startswith("Soil:")]:
                    self.add_node(soil_id, "Soil", ph=ph, organic_carbon=oc, paper=paper)
                    self.add_edge(f"Paper:{paper}", soil_id, "HAS_SOIL")
                    edges_added += 1

            rf = row.get("Rainfall")
            tmax = row.get("Temperature_Max")
            tmin = row.get("Temperature_Min")
            if pd.notna(rf) or pd.notna(tmax):
                climate_id = f"Climate:{paper}"
                if climate_id not in [n for n in self._graph.nodes() if n.startswith("Climate:")]:
                    self.add_node(climate_id, "Climate", rainfall=rf, temp_max=tmax, temp_min=tmin, paper=paper)
                    self.add_edge(f"Paper:{paper}", climate_id, "HAS_CLIMATE")
                    edges_added += 1

            gs = str(row.get("Growth_Stage", "")).strip()
            if gs and gs != "nan":
                mgmt_id = f"Management:{gs}"
                if mgmt_id not in [n for n in self._graph.nodes() if n.startswith("Management:")]:
                    self.add_node(mgmt_id, "Management", growth_stage=gs)
                for tid in [n for n in self._graph.nodes() if n.startswith("Treatment:") and paper in n]:
                    if not self._graph.has_edge(tid, mgmt_id):
                        self.add_edge(tid, mgmt_id, "MANAGES")
                        edges_added += 1

            yph = row.get("Yield_per_Hectare")
            if pd.notna(yph):
                obs_id = f"Obs:{paper}:{_}"
                self.add_node(obs_id, "Observation", yield_per_hectare=yph, paper=paper)
                if paper and f"Paper:{paper}" in self._graph:
                    self.add_edge(f"Paper:{paper}", obs_id, "OBSERVES")
                yield_id = f"Yield:{paper}:{_}"
                self.add_node(yield_id, "Yield", yield_per_hectare=yph, paper=paper)
                self.add_edge(obs_id, yield_id, "YIELDS")

        return edges_added

    def to_dict(self) -> dict:
        nodes = []
        for nid, data in self._graph.nodes(data=True):
            entry = {"id": nid, **data}
            nodes.append(entry)
        edges = []
        for u, v, data in self._graph.edges(data=True):
            entry = {"source": u, "target": v, **data}
            edges.append(entry)
        return {"nodes": nodes, "edges": edges}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save_json(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeGraph":
        data = json.loads(json_str)
        kg = cls()
        for node in data.get("nodes", []):
            nid = node.pop("id")
            ntype = node.pop("node_type", "Paper")
            kg.add_node(nid, ntype, **node)
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            etype = edge.pop("edge_type", "DESCRIBES")
            kg.add_edge(src, tgt, etype, **edge)
        return kg

    @classmethod
    def from_json_file(cls, path: str | Path) -> "KnowledgeGraph":
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_json(content)

    def page_rank(self, alpha: float = 0.85) -> dict[str, float]:
        return nx.pagerank(self._graph, alpha=alpha)

    def node_degrees(self) -> dict[str, dict]:
        result = {}
        for nid in self._graph.nodes():
            result[nid] = {
                "in_degree": self._graph.in_degree(nid),
                "out_degree": self._graph.out_degree(nid),
                "total_degree": self._graph.degree(nid),
            }
        return result
