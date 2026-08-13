"""UAASOP integration and governance tests for the Agricultural Intelligence Framework.

Each test maps one Universal Agentic AI Scientific Operating Policy requirement
(``SCIENTIFIC_AGENT_POLICY.md``) onto an AAIF mechanism and asserts the required
behavior through the project's own validation, provenance, contract, and config
layers. All tests are deterministic, offline, and fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agri_ai_agent.contracts.messages import AgentContract
from src.provenance.lineage_tracker import LineageTracker
from src.validation.range_validator import RangeValidator
from tests.governance.conftest import (
    EPISTEMIC_STATUSES,
    RecordingTaskRunner,
    build_claim,
    build_evidence,
    make_field_df,
    run_field_task,
)

CONTRACT_FILES = (
    "provenance.schema.json",
    "evidence.schema.json",
    "claim.schema.json",
    "validation.schema.json",
)

POLICY_SECTIONS = (
    "1.1 Scientific objective",
    "1.2 Evidence discipline",
    "1.3 Provenance",
    "1.4 Scientific provenance and accountability",
    "1.5 Auditability",
    "1.6 Failure preservation",
    "1.7 Layered verification",
    "1.8 Uncertainty",
    "1.9 Claim level evidence",
    "1.10 Reproducibility",
    "1.11 Human contribution",
    "1.12 Least privilege",
    "1.13 Data protection",
    "1.14 Compute discipline",
    "1.15 HPC safety",
    "1.16 Reusable knowledge",
    "1.17 Portability",
    "1.18 Predictable error behavior",
    "1.19 Conflicting evidence",
    "1.20 Completion criteria",
)

WORKFLOW_PHASES = (
    "UNDERSTAND",
    "PLAN",
    "INSPECT",
    "EXECUTE",
    "VALIDATE",
    "TRACE",
    "REVIEW",
    "REPORT",
)

REQUIRED_POLICY_FILES = (
    "SCIENTIFIC_AGENT_POLICY.md",
    "AGENT_CONFIG.yaml",
    "contracts/",
    "config/verification.yaml",
    "src/validation/",
    "src/provenance/",
)


def _load_contract_schemas(repo_root: Path) -> list[dict]:
    schemas = []
    for name in CONTRACT_FILES:
        with (repo_root / "contracts" / name).open(encoding="utf-8") as fh:
            schemas.append(json.load(fh))
    return schemas


class TestUAASOPPolicyLoading:
    def test_uaasop_policy_is_loaded(self, policy_text, agent_config, policy_sha256):
        for section in POLICY_SECTIONS:
            assert section in policy_text, f"policy missing section {section!r}"
        for phase in WORKFLOW_PHASES:
            assert phase in policy_text, f"policy missing workflow phase {phase!r}"
        assert policy_sha256 == agent_config["policy_sha256"], (
            "SCIENTIFIC_AGENT_POLICY.md no longer matches the authoritative hash "
            "recorded in AGENT_CONFIG.yaml"
        )

    def test_workflow_phases_present_in_order(self, policy_text):
        positions = [policy_text.find(phase) for phase in WORKFLOW_PHASES]
        assert all(pos != -1 for pos in positions)
        assert positions == sorted(positions)

    def test_agents_md_references_policy_infra(self, repo_root):
        agents_md = (repo_root / "AGENTS.md").read_text(encoding="utf-8")
        for reference in REQUIRED_POLICY_FILES:
            assert reference in agents_md, f"AGENTS.md must reference {reference!r}"


class TestUAASOPProvenanceRecording:
    def test_uaasop_records_tool_usage(self):
        runner = RecordingTaskRunner()
        run_field_task(make_field_df(), runner)
        assert len(runner.provenance["tools"]) >= 1
        tool_names = {t["name"] for t in runner.provenance["tools"]}
        assert "run_all_validations" in tool_names
        recorded = runner.contract.metadata["tool_uses"]
        assert recorded[-1]["name"] == "run_all_validations"

    def test_uaasop_preserves_failed_attempts(self):
        runner = RecordingTaskRunner()
        dirty_report = run_field_task(make_field_df(dirty=True), runner)
        assert dirty_report["summary"]["status"] == "FAIL"
        assert runner.complete() is False
        assert runner.contract.status == "blocked"
        assert len(runner.provenance["failures"]) >= 1

        runner.record_retry(1, "corrected out-of-range Soil_pH", "retry queued")
        clean_report = run_field_task(make_field_df(), runner)
        assert clean_report["summary"]["status"] == "PASS"
        assert runner.complete() is True
        assert runner.contract.status == "completed"
        assert runner.contract.retry_count == 1
        assert len(runner.provenance["retries"]) == 1
        assert len(runner.provenance["failures"]) >= 1

    def test_uaasop_records_human_intervention(self):
        runner = RecordingTaskRunner()
        run_field_task(make_field_df(), runner)
        runner.record_decision("release field validation report", by="agent")
        runner.record_human_intervention("approval", "human approved release of report")
        interventions = runner.provenance["human_interventions"]
        assert len(interventions) == 1
        assert interventions[0]["type"] == "approval"
        assert interventions[0]["timestamp"]
        reviewed = runner.contract.metadata["human_review"]
        assert reviewed[0]["type"] == "approval"

    def test_uaasop_records_uncertainty(self):
        runner = RecordingTaskRunner()
        runner.record_uncertainty("missing", "Yield for treatment X not recorded in source PDF")
        entries = runner.provenance["uncertainty"]
        assert len(entries) == 1
        assert entries[0]["status"] in EPISTEMIC_STATUSES
        assert "value" not in entries[0], "uncertainty must not fabricate a value"
        assert runner.contract.metadata["uncertainty"][0]["status"] == "missing"


class TestUAASOPVerification:
    def test_uaasop_invokes_validation(self):
        runner = RecordingTaskRunner()
        report = run_field_task(make_field_df(), runner)
        summary = report["summary"]
        assert summary["status"] == "PASS"
        assert summary["total_checks"] >= 2
        validator_names = {r["validator"] for r in report["results"]}
        assert {"RangeValidator", "DataQualityValidator"} <= validator_names
        assert runner.provenance["validations"][-1]["result"] == "passed"

    def test_uaasop_blocks_completion_after_validation_failure(self):
        runner = RecordingTaskRunner()
        report = run_field_task(make_field_df(dirty=True), runner)
        assert report["summary"]["status"] == "FAIL"
        range_result = next(r for r in report["results"] if r["validator"] == "RangeValidator")
        assert range_result["passed"] is False
        assert runner.complete() is False
        assert runner.contract.status == "blocked"
        assert len(runner.contract.errors) >= 1


class TestUAASOPEvidenceDiscipline:
    def test_uaasop_separates_evidence_and_inference(self):
        observed = build_evidence(
            evidence_id="ev-obs-1",
            content="Measured soil pH 6.5 in field log",
            status="observed",
            source_reference="doi:10.0000/field-log",
        )
        inferred = build_evidence(
            evidence_id="ev-inf-1",
            content="Soil pH trend inferred from repeated readings",
            status="inferred",
            source_reference="doi:10.0000/field-log",
        )
        assert observed["status"] == "observed"
        assert inferred["status"] == "inferred"
        assert inferred["status"] != "observed", "inference must not be relabelled as fact"

        claim = build_claim(
            claim_id="cl-1",
            claim="pH trend is increasing",
            status="inferred",
            evidence=["ev-obs-1", "ev-inf-1"],
            sources=["doi:10.0000/field-log"],
        )
        assert claim["status"] == "inferred"
        assert claim["evidence"] == ["ev-obs-1", "ev-inf-1"]

    def test_uaasop_uncertainty_is_labeled_not_fabricated(self):
        missing = build_evidence(
            evidence_id="ev-miss-1",
            content="Nitrogen value absent from source table",
            status="missing",
            source_reference="doi:10.0000/field-log",
        )
        assert missing["status"] == "missing"
        assert "value" not in missing


class TestUAASOPResourceLimits:
    def test_uaasop_respects_resource_limits(self, connector_limits):
        defaults = connector_limits["defaults"]
        assert defaults["rate_limit_per_minute"] > 0
        assert defaults["timeout_sec"] > 0
        assert defaults["retry_max"] >= 0
        assert defaults["retry_base_delay_sec"] >= 0
        for group in ("literature", "agricultural"):
            for name, cfg in connector_limits[group].items():
                rate = cfg["rate_limit_per_minute"]
                assert rate > 0, f"{name} rate limit must be positive"
                assert rate <= 200, f"{name} rate limit looks unreasonable: {rate}"
                assert cfg.get("timeout_sec", defaults["timeout_sec"]) > 0
                assert cfg.get("retry_max", defaults["retry_max"]) >= 0
        assert connector_limits["literature"]["Crossref"]["rate_limit_per_minute"] == 50


class TestUAASOPAuditability:
    def test_uaasop_provenance_is_auditable(self, tmp_path):
        runner = RecordingTaskRunner()
        run_field_task(make_field_df(), runner)
        prov = runner.provenance
        for key in ("tools", "inputs", "sources", "transformations", "outputs"):
            assert prov[key], f"audit chain link {key!r} is empty"

        lineage_dir = tmp_path / "lineage"
        tracker = LineageTracker(lineage_dir=lineage_dir)
        dataset_id = tracker.initialize_dataset("field_2024", {"source": "doi:10.0000/x"})
        raw = lineage_dir / "raw.csv"
        raw.write_text("raw data")
        tracker.transition(dataset_id, "api_source", "raw_dataset", raw)
        clean = lineage_dir / "clean.csv"
        clean.write_text("clean data")
        tracker.transition(dataset_id, "raw_dataset", "clean_dataset", clean)

        lineage = tracker.get_lineage(dataset_id)
        assert [r["to_stage"] for r in lineage] == ["api_source", "raw_dataset", "clean_dataset"]
        assert all(r["artifact_path"] for r in lineage[1:])
        assert all(r["timestamp"] for r in lineage)


class TestUAASOPGovernanceIntegrity:
    def test_agent_config_parses_and_scientific_mode(self, agent_config):
        assert agent_config["policy_version"] == "0.1.0"
        assert agent_config["scientific_mode"] is True
        assert agent_config["policy_source"] == "SCIENTIFIC_AGENT_POLICY.md"
        for key in (
            "provenance",
            "verification",
            "uncertainty",
            "reproducibility",
            "human_review",
            "security",
            "compute",
        ):
            assert key in agent_config, f"AGENT_CONFIG.yaml missing section {key!r}"

    def test_contracts_are_draft2020_structured(self, repo_root):
        schemas = _load_contract_schemas(repo_root)
        assert len(schemas) == 4
        for schema in schemas:
            assert "$schema" in schema
            assert schema["$schema"].startswith("https://json-schema.org/draft/2020-12/"), schema[
                "$schema"
            ]
            assert schema["type"] == "object"
            assert schema["required"]
            for required in schema["required"]:
                assert required in schema["properties"]

    def test_contracts_validate_real_instances(self, repo_root):
        jsonschema = pytest.importorskip("jsonschema")
        schemas = {s["title"]: s for s in _load_contract_schemas(repo_root)}
        evidence = build_evidence(
            evidence_id="ev-1",
            content="measured pH 6.5",
            status="observed",
            source_reference="doi:10.0000/x",
        )
        claim = build_claim(
            claim_id="cl-1", claim="yield differs", status="inferred", evidence=["ev-1"]
        )
        runner = RecordingTaskRunner()
        run_field_task(make_field_df(), runner)
        instances = {
            "Scientific Evidence Record": evidence,
            "Scientific Claim Record": claim,
            "Scientific Provenance Record": runner.provenance,
            "Scientific Validation Record": {
                "id": "val-1",
                "target": "field_table",
                "type": "domain",
                "result": "passed",
                "timestamp": runner.provenance["created_at"],
            },
        }
        for title, instance in instances.items():
            schema = schemas[title]
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.validate(instance, schema)

    def test_contracts_reject_invalid_status(self, repo_root):
        jsonschema = pytest.importorskip("jsonschema")
        schema = next(
            s
            for s in _load_contract_schemas(repo_root)
            if s["title"] == "Scientific Evidence Record"
        )
        bad = build_evidence(
            evidence_id="ev-bad",
            content="fabricated",
            status="fabricated",
            source_reference="doi:10.0000/x",
        )
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)

    def test_api_keys_env_contains_no_secrets(self, repo_root):
        env_path = repo_root / "config" / "api_keys.env"
        assert env_path.exists()
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            assert key.strip(), "empty key in api_keys.env"
            assert value.strip().strip('"').strip("'") == "", (
                f"{key.strip()} must not ship a non-empty secret value"
            )


class TestUAASOPAgentContract:
    def test_agent_contract_round_trip(self):
        contract = AgentContract(agent_name="TestResearchAgent", status="completed")
        contract.errors.append("recorded failure")
        restored = AgentContract.from_dict(contract.to_dict())
        assert restored.agent_name == "TestResearchAgent"
        assert restored.errors == ["recorded failure"]

    def test_range_validator_detects_out_of_range(self):
        df = make_field_df(dirty=True)
        result = RangeValidator().validate(df)
        assert result["Soil_pH"]["out_of_range_count"] >= 1
        clean = RangeValidator().validate(make_field_df())
        assert clean["Soil_pH"]["out_of_range_count"] == 0
