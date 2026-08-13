"""Fixtures and harness for the UAASOP integration (governance) test suite.

Everything here is deterministic and offline: no LLM calls, no network, no
paid compute. The harness models a minimal policy-compliant agent task run
(EXECUTE -> VALIDATE -> TRACE -> REVIEW) and records a provenance trace that
mirrors ``AGENT_CONFIG.yaml`` settings and the UAASOP provenance fields from
``SCIENTIFIC_AGENT_POLICY.md`` section 3.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml

from agri_ai_agent.contracts.messages import AgentContract
from src.validation.reports import run_all_validations

# Epistemic statuses defined by policy section 4.
EPISTEMIC_STATUSES = {
    "verified",
    "observed",
    "retrieved",
    "calculated",
    "estimated",
    "inferred",
    "predicted",
    "assumed",
    "unknown",
    "missing",
    "conflicting",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_field_df(dirty: bool = False) -> pd.DataFrame:
    """Return a small deterministic UAMS-shaped field table.

    ``dirty=True`` injects an out-of-range ``Soil_pH`` value so the range
    validation layer reports a violation.
    """
    ph = [6.5, 7.2, 6.0, 6.8, 7.0, 5.9]
    if dirty:
        ph[0] = 15.0
    return pd.DataFrame(
        {
            "Crop": ["Rice", "Wheat", "Maize", "Rice", "Wheat", "Maize"],
            "Soil_pH": ph,
            "Rainfall": [1200, 650, 800, 1100, 700, 850],
            "Nitrogen": [90, 120, 100, 95, 110, 105],
            "Yield_per_Hectare": [4.2, 3.1, 5.0, 4.5, 3.3, 5.2],
        }
    )


def build_evidence(
    evidence_id: str,
    content: str,
    status: str,
    source_reference: str,
    source_type: str = "dataset",
    confidence_level: str = "high",
) -> dict:
    """Return a dict shaped for ``contracts/evidence.schema.json``."""
    return {
        "id": evidence_id,
        "content": content,
        "status": status,
        "source": {
            "reference": source_reference,
            "type": source_type,
            "accessed_at": _now(),
        },
        "confidence": {
            "level": confidence_level,
            "basis": "recorded from source, not inferred",
        },
        "recorded_by": "TestResearchAgent",
    }


def build_claim(
    claim_id: str,
    claim: str,
    status: str,
    evidence: list[str] | None = None,
    sources: list[str] | None = None,
    consequential: bool = False,
) -> dict:
    """Return a dict shaped for ``contracts/claim.schema.json``."""
    return {
        "id": claim_id,
        "claim": claim,
        "status": status,
        "evidence": evidence or [],
        "sources": sources or [],
        "consequential": consequential,
        "confidence": {"level": "medium", "basis": "single source"},
    }


class RecordingTaskRunner:
    """Deterministic harness simulating one policy-compliant agent task run.

    The runner records tools, inputs, sources, transformations, outputs,
    decisions, failures, retries, validations, human interventions, and
    uncertainty into both a UAASOP-shaped provenance dict and an
    ``AgentContract``. Completion is blocked while the most recent validation
    result is anything other than ``passed`` (policy section 1.20), while
    earlier failures stay preserved in the record (policy section 1.6).
    """

    def __init__(self, agent_name: str = "TestResearchAgent"):
        self.agent_name = agent_name
        self.contract = AgentContract(agent_name=agent_name, status="pending")
        self._last_validation_result: str | None = None
        self.provenance: dict = {
            "id": f"prov-{agent_name.lower().replace(' ', '-')}",
            "task": "Validate field measurements before release",
            "created_at": _now(),
            "agent": {"name": agent_name},
            "tools": [],
            "inputs": [],
            "sources": [],
            "transformations": [],
            "outputs": [],
            "decisions": [],
            "assumptions": [],
            "failures": [],
            "retries": [],
            "validations": [],
            "human_interventions": [],
            "uncertainty": [],
        }

    def start(self) -> None:
        self.contract.started_at = datetime.now(timezone.utc)

    def use_tool(self, name: str, version: str = "", command: str = "") -> None:
        entry = {"name": name, "version": version, "command": command}
        self.provenance["tools"].append(entry)
        self.contract.metadata.setdefault("tool_uses", []).append(entry)

    def record_input(self, reference: str, description: str = "") -> None:
        entry = {"reference": reference, "description": description}
        self.provenance["inputs"].append(entry)
        self.contract.input_data["reference"] = reference

    def record_source(self, reference: str, source_type: str = "dataset") -> None:
        self.provenance["sources"].append({"reference": reference, "type": source_type})

    def record_transformation(self, description: str, method: str) -> None:
        self.provenance["transformations"].append({"description": description, "method": method})

    def record_failure(self, description: str, cause: str, outcome: str) -> None:
        self.provenance["failures"].append(
            {"description": description, "cause": cause, "outcome": outcome}
        )
        self.contract.errors.append(description)

    def record_retry(self, attempt: int, change: str, outcome: str) -> None:
        self.provenance["retries"].append(
            {"attempt": attempt, "change": change, "outcome": outcome}
        )
        self.contract.retry_count += 1

    def record_validation(self, validation_type: str, result: str, details: str) -> None:
        self.provenance["validations"].append(
            {"type": validation_type, "result": result, "details": details}
        )
        self._last_validation_result = result

    def record_human_intervention(self, intervention_type: str, description: str) -> None:
        entry = {"type": intervention_type, "description": description, "timestamp": _now()}
        self.provenance["human_interventions"].append(entry)
        self.contract.metadata.setdefault("human_review", []).append(entry)

    def record_decision(self, description: str, by: str = "agent") -> None:
        self.provenance["decisions"].append(
            {"description": description, "by": by, "timestamp": _now()}
        )

    def record_output(self, reference: str, description: str = "") -> None:
        entry = {"reference": reference, "description": description}
        self.provenance["outputs"].append(entry)
        self.contract.output_data["reference"] = reference

    def record_uncertainty(self, status: str, description: str) -> None:
        entry = {"status": status, "description": description, "timestamp": _now()}
        self.provenance.setdefault("uncertainty", []).append(entry)
        self.contract.metadata.setdefault("uncertainty", []).append(entry)

    def complete(self, force: bool = False) -> bool:
        """Close the run, blocking completion unless validation passed.

        Returns True when the task may be declared complete. Past validation
        failures are preserved in ``self.provenance`` but only the most recent
        validation result gates completion.
        """
        if not force and self._last_validation_result is not None:
            if self._last_validation_result != "passed":
                self.contract.status = "blocked"
                return False
        self.contract.completed_at = datetime.now(timezone.utc)
        self.contract.status = "completed"
        return True


def run_field_task(df: pd.DataFrame, runner: RecordingTaskRunner) -> dict:
    """Execute the field-validation task against the project's own validators.

    Uses ``src.validation.reports.run_all_validations`` (layered verification
    per policy section 1.7) and records the outcome into ``runner``.
    """
    runner.start()
    runner.use_tool("run_all_validations", version="", command="validate(df)")
    runner.record_input(reference="uams:field_table", description="UAMS field table")
    runner.record_source(reference="doi:10.0000/test-field-data", source_type="dataset")
    runner.record_transformation("validate ranges, quality, provenance", "layered validation")

    report = run_all_validations(df)
    summary = report["summary"]
    result = "passed" if summary["status"] == "PASS" else "failed"
    runner.record_validation(
        "domain",
        result,
        f"layered validation status={summary['status']}",
    )
    runner.record_output("validation_report.json", "field-level validation report")
    if summary["status"] == "FAIL":
        runner.record_failure(
            "Layered validation failed",
            f"{summary['failed']} of {summary['total_checks']} checks failed",
            "validation FAIL",
        )
    return report


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the AAIF repository root."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def policy_text(repo_root) -> str:
    return (repo_root / "SCIENTIFIC_AGENT_POLICY.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def policy_sha256(policy_text) -> str:
    return hashlib.sha256(policy_text.encode("utf-8")).hexdigest()


@pytest.fixture(scope="session")
def agent_config(repo_root) -> dict:
    with (repo_root / "AGENT_CONFIG.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="session")
def connector_limits(repo_root) -> dict:
    with (repo_root / "config" / "connector_limits.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
