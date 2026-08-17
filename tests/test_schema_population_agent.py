"""Tests for Schema Population Agent."""

import numpy as np
import pandas as pd

from agri_ai_agent.agents.schema_population_agent import SchemaPopulationAgent
from agri_ai_agent.contracts.messages import AgentContract


class TestSchemaPopulationAgent:
    def setup_method(self):
        self.agent = SchemaPopulationAgent()

    def test_agent_name(self):
        assert self.agent.agent_name == "SchemaPopulationAgent"

    def test_process_empty_dataframe(self):
        result = self.agent.process(pd.DataFrame())
        assert result.empty

    def test_yield_plot_to_hectare(self):
        df = pd.DataFrame(
            {
                "Yield_per_Plot": [5.0, 6.0],
                "Plot_Size": [20.0, 20.0],
            }
        )
        result = self.agent.process(df)
        assert "Yield_per_Hectare" in result.columns
        expected = [5.0 * (10000 / 20), 6.0 * (10000 / 20)]
        np.testing.assert_array_almost_equal(
            result["Yield_per_Hectare"].values, expected, decimal=2
        )

    def test_yield_acre_to_hectare(self):
        df = pd.DataFrame({"Yield_per_Acre": [1000.0, 1500.0]})
        result = self.agent.process(df)
        assert "Yield_per_Hectare" in result.columns
        expected = [1000.0 * 2.471, 1500.0 * 2.471]
        np.testing.assert_array_almost_equal(
            result["Yield_per_Hectare"].values, expected, decimal=2
        )

    def test_organic_matter_from_carbon(self):
        df = pd.DataFrame({"Organic_Carbon": [0.5, 1.0]})
        result = self.agent.process(df)
        assert "Organic_Matter" in result.columns
        expected = [0.5 * 1.724, 1.0 * 1.724]
        np.testing.assert_array_almost_equal(result["Organic_Matter"].values, expected, decimal=3)

    def test_organic_carbon_from_matter(self):
        df = pd.DataFrame({"Organic_Matter": [0.862, 1.724]})
        result = self.agent.process(df)
        assert "Organic_Carbon" in result.columns
        expected = [0.862 / 1.724, 1.724 / 1.724]
        np.testing.assert_array_almost_equal(result["Organic_Carbon"].values, expected, decimal=4)

    def test_biomass_from_shoot_root(self):
        df = pd.DataFrame(
            {
                "Shoot_Biomass_g": [25.0, 30.0],
                "Root_Biomass_g": [5.0, 7.0],
            }
        )
        result = self.agent.process(df)
        assert "Biomass_Yield" in result.columns
        expected = [30.0, 37.0]
        np.testing.assert_array_almost_equal(result["Biomass_Yield"].values, expected, decimal=2)

    def test_gdd_calculation(self):
        df = pd.DataFrame(
            {
                "Average_Temperature": [25.0, 20.0],
                "Growth_Duration_Days": [90, 120],
            }
        )
        result = self.agent.process(df)
        assert "Growing_Degree_Days" in result.columns
        expected = [(25 - 10) * 90, (20 - 10) * 120]
        np.testing.assert_array_almost_equal(
            result["Growing_Degree_Days"].values, expected, decimal=2
        )

    def test_nue_is_not_derived_to_avoid_leakage(self):
        """NUE = Yield/N is yield-derived; it must not be created as an ML feature."""
        df = pd.DataFrame(
            {
                "Yield_per_Hectare": [4000.0, 5000.0],
                "Nitrogen": [120.0, 150.0],
            }
        )
        result = self.agent.process(df)
        assert "Nitrogen_Use_Efficiency" not in result.columns

    def test_nx_p_interaction(self):
        df = pd.DataFrame({"Nitrogen": [100, 150], "Phosphorus": [40, 60]})
        result = self.agent.process(df)
        assert "N_x_P" in result.columns
        expected = [4000, 9000]
        np.testing.assert_array_almost_equal(result["N_x_P"].values, expected, decimal=2)

    def test_does_not_overwrite_existing(self):
        df = pd.DataFrame(
            {
                "Yield_per_Plot": [5.0, 6.0],
                "Plot_Size": [20.0, 20.0],
                "Yield_per_Hectare": [3000.0, 3500.0],
            }
        )
        result = self.agent.process(df)
        expected = [3000.0, 3500.0]
        np.testing.assert_array_almost_equal(
            result["Yield_per_Hectare"].values, expected, decimal=2
        )

    def test_skips_rows_with_missing_sources(self):
        df = pd.DataFrame(
            {
                "Yield_per_Plot": [5.0, np.nan],
                "Plot_Size": [20.0, 20.0],
            }
        )
        result = self.agent.process(df)
        assert result.iloc[0]["Yield_per_Hectare"] == 2500.0
        assert pd.isna(result.iloc[1]["Yield_per_Hectare"])

    def test_population_report_structure(self):
        df = pd.DataFrame(
            {
                "Yield_per_Plot": [5.0],
                "Plot_Size": [20.0],
                "Organic_Carbon": [0.5],
            }
        )
        self.agent.process(df)
        report = self.agent.population_report
        assert "rules_evaluated" in report
        assert "rules_applied" in report
        assert "derivations" in report
        assert report["rules_applied"] >= 2

    def test_run_method_returns_contract(self):
        df = pd.DataFrame({"Yield_per_Plot": [5.0], "Plot_Size": [20.0]})
        contract = self.agent.run(df)
        assert isinstance(contract, AgentContract)
        assert contract.status == "success"
        assert "population_report" in contract.output_data

    def test_custom_min_confidence(self):
        df = pd.DataFrame(
            {
                "Yield_per_Plot": [5.0],
                "Plot_Size": [20.0],
            }
        )
        result = self.agent.process(df, min_confidence=1.0)
        assert pd.isna(result.iloc[0].get("Yield_per_Hectare", np.nan))

    def test_multiple_derivation_paths(self):
        df = pd.DataFrame(
            {
                "Yield_per_Plot": [5.0],
                "Plot_Size": [20.0],
                "Organic_Carbon": [0.5],
                "Shoot_Biomass_g": [25.0],
                "Root_Biomass_g": [5.0],
                "Average_Temperature": [25.0],
                "Growth_Duration_Days": [90],
            }
        )
        result = self.agent.process(df)
        assert "Yield_per_Hectare" in result.columns
        assert "Organic_Matter" in result.columns
        assert "Biomass_Yield" in result.columns
        assert "Growing_Degree_Days" in result.columns
