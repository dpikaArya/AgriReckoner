"""Tests for KnowledgeGraph — Phase 13."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_ai_agent.knowledge_graph.graph import (
    KnowledgeGraph,
    VALID_NODE_TYPES,
    VALID_EDGE_TYPES,
    EDGE_NODE_PAIRS,
)


@pytest.fixture
def kg():
    return KnowledgeGraph()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Source_Paper": ["Paper_A", "Paper_A", "Paper_B"],
        "Crop": ["Rice", "Rice", "Wheat"],
        "Fertilizer_Name": ["Urea", "DAP", "NPK"],
        "Dose": [50.0, 30.0, 40.0],
        "Application_Interval": [30, 45, 60],
        "Soil_pH": [6.5, np.nan, 7.0],
        "Organic_Carbon": [0.8, np.nan, 0.5],
        "Rainfall": [1200, np.nan, 800],
        "Temperature_Max": [33, np.nan, 30],
        "Temperature_Min": [22, np.nan, 18],
        "Growth_Stage": ["Flowering", "Flowering", "Vegetative"],
        "Yield_per_Hectare": [4.5, 3.8, 3.2],
        "Year": [2022, 2022, 2021],
        "Country": ["India", "India", "USA"],
        "DOI": ["10.1000/a", "10.1000/a", "10.2000/b"],
    })


class TestKnowledgeGraph:

    def test_initial_state(self, kg):
        assert kg.node_count == 0
        assert kg.edge_count == 0
        assert isinstance(kg.graph, type(kg.graph))

    def test_add_node(self, kg):
        kg.add_node("Paper:1", "Paper", title="Test")
        assert kg.node_count == 1
        node = kg.get_node("Paper:1")
        assert node["node_type"] == "Paper"
        assert node["title"] == "Test"

    def test_add_node_invalid_type(self, kg):
        with pytest.raises(ValueError, match="Invalid node type"):
            kg.add_node("X:1", "InvalidType")

    def test_add_edge(self, kg):
        kg.add_node("Paper:1", "Paper")
        kg.add_node("Crop:Rice", "Crop")
        kg.add_edge("Paper:1", "Crop:Rice", "TREATS")
        assert kg.edge_count == 1

    def test_add_edge_invalid_type(self, kg):
        kg.add_node("Paper:1", "Paper")
        kg.add_node("Crop:Rice", "Crop")
        with pytest.raises(ValueError, match="Invalid edge type"):
            kg.add_edge("Paper:1", "Crop:Rice", "INVALID_EDGE")

    def test_add_edge_wrong_source_type(self, kg):
        kg.add_node("Crop:Rice", "Crop")
        kg.add_node("Paper:1", "Paper")
        with pytest.raises(ValueError, match="source of type"):
            kg.add_edge("Crop:Rice", "Paper:1", "DESCRIBES")

    def test_add_edge_wrong_target_type(self, kg):
        kg.add_node("Paper:1", "Paper")
        kg.add_node("Crop:Rice", "Crop")
        with pytest.raises(ValueError, match="target of type"):
            kg.add_edge("Paper:1", "Crop:Rice", "DESCRIBES")

    def test_get_node_nonexistent(self, kg):
        assert kg.get_node("nonexistent") is None

    def test_get_nodes_by_type(self, kg):
        kg.add_node("P1", "Paper", title="A")
        kg.add_node("P2", "Paper", title="B")
        kg.add_node("C1", "Crop", name="Rice")
        papers = kg.get_nodes_by_type("Paper")
        assert len(papers) == 2
        crops = kg.get_nodes_by_type("Crop")
        assert len(crops) == 1
        assert crops[0]["name"] == "Rice"

    def test_get_edges_by_type(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        kg.add_node("C2", "Crop")
        kg.add_edge("P1", "C1", "TREATS")
        kg.add_edge("P1", "C2", "TREATS")
        edges = kg.get_edges_by_type("TREATS")
        assert len(edges) == 2

    def test_get_neighbors(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        kg.add_node("S1", "Soil")
        kg.add_edge("P1", "C1", "TREATS")
        kg.add_edge("P1", "S1", "HAS_SOIL")
        out = kg.get_neighbors("P1", "out")
        assert "C1" in out
        assert "S1" in out
        assert len(out) == 2
        assert kg.get_neighbors("C1", "in") == ["P1"]
        assert kg.get_neighbors("C1", "out") == []
        both = kg.get_neighbors("P1", "both")
        assert len(both) == 2

    def test_get_neighbors_nonexistent(self, kg):
        assert kg.get_neighbors("nope") == []

    def test_shortest_path_found(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        kg.add_node("T1", "Treatment")
        kg.add_edge("P1", "C1", "TREATS")
        kg.add_edge("C1", "T1", "GROWS")
        path = kg.shortest_path("P1", "T1")
        assert path == ["P1", "C1", "T1"]

    def test_shortest_path_not_found(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        assert kg.shortest_path("P1", "C1") is None

    def test_centrality_degree(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        kg.add_node("S1", "Soil")
        kg.add_edge("P1", "C1", "TREATS")
        kg.add_edge("P1", "S1", "HAS_SOIL")
        cent = kg.centrality("degree")
        assert cent["P1"] > cent["C1"]

    def test_centrality_betweenness(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        kg.add_node("T1", "Treatment")
        kg.add_edge("P1", "C1", "TREATS")
        kg.add_edge("C1", "T1", "GROWS")
        cent = kg.centrality("betweenness")
        assert cent["C1"] > 0

    def test_centrality_invalid_method(self, kg):
        with pytest.raises(ValueError, match="Unknown centrality"):
            kg.centrality("invalid")

    def test_summary_empty(self, kg):
        s = kg.summary()
        assert s["total_nodes"] == 0
        assert s["total_edges"] == 0
        assert s["density"] == 0.0

    def test_summary_populated(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        kg.add_edge("P1", "C1", "TREATS")
        s = kg.summary()
        assert s["total_nodes"] == 2
        assert s["total_edges"] == 1
        assert s["node_type_counts"]["Paper"] == 1
        assert s["edge_type_counts"]["TREATS"] == 1

    def test_build_from_dataframe(self, kg, sample_df):
        edges = kg.build_from_dataframe(sample_df)
        assert kg.node_count > 0
        assert kg.edge_count > 0
        papers = kg.get_nodes_by_type("Paper")
        assert len(papers) >= 2
        crops = kg.get_nodes_by_type("Crop")
        assert len(crops) >= 2

    def test_build_from_dataframe_has_soil_climate(self, kg, sample_df):
        kg.build_from_dataframe(sample_df)
        soils = kg.get_nodes_by_type("Soil")
        climates = kg.get_nodes_by_type("Climate")
        assert len(soils) >= 1
        assert len(climates) >= 1

    def test_build_from_dataframe_has_treatments(self, kg, sample_df):
        kg.build_from_dataframe(sample_df)
        treatments = kg.get_nodes_by_type("Treatment")
        assert len(treatments) >= 3

    def test_build_from_dataframe_has_management(self, kg, sample_df):
        kg.build_from_dataframe(sample_df)
        mgmt = kg.get_nodes_by_type("Management")
        assert len(mgmt) >= 2

    def test_build_from_dataframe_has_observations_yields(self, kg, sample_df):
        kg.build_from_dataframe(sample_df)
        obs = kg.get_nodes_by_type("Observation")
        yld = kg.get_nodes_by_type("Yield")
        assert len(obs) == 3
        assert len(yld) == 3

    def test_to_dict(self, kg):
        kg.add_node("P1", "Paper", title="Test")
        kg.add_node("C1", "Crop", name="Rice")
        kg.add_edge("P1", "C1", "TREATS")
        d = kg.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["nodes"][0]["id"] == "P1"

    def test_to_json_and_back(self, kg):
        kg.add_node("P1", "Paper", title="Test")
        kg.add_node("C1", "Crop", name="Rice")
        kg.add_edge("P1", "C1", "TREATS")
        j = kg.to_json()
        kg2 = KnowledgeGraph.from_json(j)
        assert kg2.node_count == 2
        assert kg2.edge_count == 1
        assert kg2.get_node("P1")["title"] == "Test"

    def test_save_and_load_json(self, kg, tmp_path):
        kg.add_node("P1", "Paper", title="X")
        kg.add_node("C1", "Crop", name="Y")
        kg.add_edge("P1", "C1", "TREATS")
        path = tmp_path / "kg.json"
        kg.save_json(path)
        assert path.exists()
        kg2 = KnowledgeGraph.from_json_file(path)
        assert kg2.node_count == 2

    def test_page_rank(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        kg.add_node("S1", "Soil")
        kg.add_edge("P1", "C1", "TREATS")
        kg.add_edge("P1", "S1", "HAS_SOIL")
        pr = kg.page_rank()
        assert "P1" in pr
        assert abs(sum(pr.values()) - 1.0) < 0.01

    def test_node_degrees(self, kg):
        kg.add_node("P1", "Paper")
        kg.add_node("C1", "Crop")
        kg.add_edge("P1", "C1", "TREATS")
        deg = kg.node_degrees()
        assert deg["P1"]["out_degree"] == 1
        assert deg["P1"]["in_degree"] == 0
        assert deg["C1"]["in_degree"] == 1

    def test_empty_dataframe(self, kg):
        df = pd.DataFrame({"Source_Paper": [], "Crop": []})
        edges = kg.build_from_dataframe(df)
        assert kg.node_count == 0
        assert edges == 0

    def test_valid_node_types_count(self):
        assert len(VALID_NODE_TYPES) == 12

    def test_valid_edge_types_count(self):
        assert len(VALID_EDGE_TYPES) == 12

    def test_edge_node_pairs_count(self):
        assert len(EDGE_NODE_PAIRS) == 12
