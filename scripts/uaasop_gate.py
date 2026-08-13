"""UAASOP integration quality gate for the Agricultural Intelligence Framework.

Runs the static policy-infrastructure checks that the CI pipeline must enforce
before (and independently of) the behavioral governance tests:

1. ``SCIENTIFIC_AGENT_POLICY.md`` matches the authoritative sha256 recorded in
   ``AGENT_CONFIG.yaml`` (``policy_sha256``).
2. ``AGENT_CONFIG.yaml`` parses and enables the required policy sections.
3. ``contracts/*.json`` are valid JSON Schema Draft 2020-12 documents with the
   required fields and policy-defined enums.
4. ``AGENTS.md`` references the policy infrastructure files.
5. ``config/api_keys.env`` contains no non-empty secret values.
6. ``config/verification.yaml`` and ``config/connector_limits.yaml`` parse and
   carry the thresholds/limits the policy requires.

Exit code 0 means every check passed; 1 means at least one failed. Uses only
the Python standard library plus PyYAML (a project runtime dependency).

Usage: python scripts/uaasop_gate.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

POLICY_FILE = "SCIENTIFIC_AGENT_POLICY.md"
AGENT_CONFIG_FILE = "AGENT_CONFIG.yaml"
AGENTS_MD_FILE = "AGENTS.md"
CONTRACT_FILES = (
    "provenance.schema.json",
    "evidence.schema.json",
    "claim.schema.json",
    "validation.schema.json",
)

# Policy section 4 epistemic statuses.
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

# Policy section 1.7 verification layers.
VALIDATION_TYPES = {
    "schema",
    "deterministic",
    "mathematical",
    "statistical",
    "domain",
    "cross_source",
    "independent",
    "human_expert",
}

REQUIRED_AGENT_CONFIG_SECTIONS = (
    "provenance",
    "verification",
    "uncertainty",
    "reproducibility",
    "human_review",
    "security",
    "compute",
)

AGENTS_MD_REFERENCES = (
    "SCIENTIFIC_AGENT_POLICY.md",
    "AGENT_CONFIG.yaml",
    "contracts/",
    "config/verification.yaml",
    "src/validation/",
    "src/provenance/",
)

_CHECKS: list[tuple[str, str]] = []


def record(check: str, outcome: str) -> None:
    _CHECKS.append((check, outcome))


def fail(check: str, message: str) -> None:
    record(check, f"FAIL: {message}")


def ok(check: str) -> None:
    record(check, "PASS")


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a YAML mapping")
    return data


def check_policy_integrity() -> None:
    policy = REPO_ROOT / POLICY_FILE
    if not policy.is_file():
        fail("policy-integrity", f"{POLICY_FILE} is missing")
        return
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    config = load_yaml(REPO_ROOT / AGENT_CONFIG_FILE)
    expected = config.get("policy_sha256")
    if expected == digest:
        ok("policy-integrity")
    else:
        fail(
            "policy-integrity",
            f"sha256 {digest} != AGENT_CONFIG policy_sha256 {expected}",
        )


def check_agent_config() -> None:
    try:
        config = load_yaml(REPO_ROOT / AGENT_CONFIG_FILE)
    except Exception as exc:  # noqa: BLE001 - gate must report the root cause
        fail("agent-config", f"unparsable: {exc}")
        return
    if config.get("policy_version") != "0.1.0":
        fail("agent-config", "policy_version must be 0.1.0")
        return
    if config.get("scientific_mode") is not True:
        fail("agent-config", "scientific_mode must be true")
        return
    missing = [k for k in REQUIRED_AGENT_CONFIG_SECTIONS if k not in config]
    if missing:
        fail("agent-config", f"missing sections: {', '.join(missing)}")
        return
    ok("agent-config")


def check_contracts() -> None:
    contracts_dir = REPO_ROOT / "contracts"
    all_valid = True
    for name in CONTRACT_FILES:
        check_name = f"contracts/{name}"
        path = contracts_dir / name
        if not path.is_file():
            fail(check_name, "missing contract file")
            all_valid = False
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                schema = json.load(fh)
        except json.JSONDecodeError as exc:
            fail(check_name, f"not valid JSON: {exc}")
            all_valid = False
            continue
        if not schema.get("$schema", "").startswith("https://json-schema.org/draft/2020-12/"):
            fail(check_name, "must be JSON Schema Draft 2020-12")
            all_valid = False
            continue
        if schema.get("type") != "object" or not schema.get("required"):
            fail(check_name, "must be an object with a non-empty required list")
            all_valid = False
            continue
        if any(r not in schema.get("properties", {}) for r in schema["required"]):
            fail(check_name, "some required fields are not defined in properties")
            all_valid = False
            continue
        title = schema.get("title", "")
        if title == "Scientific Evidence Record" or title == "Scientific Claim Record":
            status_values = set(schema["properties"]["status"].get("enum") or [])
            if not status_values or not status_values <= EPISTEMIC_STATUSES:
                fail(check_name, "status enum must be exactly the policy section 4 set")
                all_valid = False
                continue
        if title == "Scientific Validation Record":
            type_values = set(schema["properties"]["type"].get("enum") or [])
            if not type_values or not type_values <= VALIDATION_TYPES:
                fail(check_name, "type enum must be a policy section 1.7 verification layer")
                all_valid = False
                continue
        ok(check_name)
    if all_valid:
        ok("contracts")


def check_agents_md() -> None:
    agents_md = REPO_ROOT / AGENTS_MD_FILE
    if not agents_md.is_file():
        fail("agents-md", f"{AGENTS_MD_FILE} is missing")
        return
    text = agents_md.read_text(encoding="utf-8")
    missing = [ref for ref in AGENTS_MD_REFERENCES if ref not in text]
    if missing:
        fail("agents-md", f"must reference {', '.join(missing)}")
        return
    ok("agents-md")


def check_secrets() -> None:
    env_path = REPO_ROOT / "config" / "api_keys.env"
    if not env_path.is_file():
        fail("secrets", "config/api_keys.env is missing")
        return
    offenders = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            offenders.append(f"{key.strip()}={value}")
    if offenders:
        fail("secrets", f"non-empty secret values present: {', '.join(offenders)}")
        return
    ok("secrets")


def check_verification_config() -> None:
    try:
        config = load_yaml(REPO_ROOT / "config" / "verification.yaml")
    except Exception as exc:  # noqa: BLE001 - gate must report the root cause
        fail("verification-config", f"unparsable: {exc}")
        return
    chain = (config.get("verification") or {}).get("chain") or {}
    if not chain.get("required_links"):
        fail("verification-config", "verification.chain.required_links is empty")
        return
    completeness = chain.get("min_completeness")
    if not isinstance(completeness, (int, float)) or not 0 < completeness <= 1:
        fail("verification-config", "min_completeness must be in (0, 1]")
        return
    ok("verification-config")


def check_connector_limits() -> None:
    try:
        limits = load_yaml(REPO_ROOT / "config" / "connector_limits.yaml")
    except Exception as exc:  # noqa: BLE001 - gate must report the root cause
        fail("connector-limits", f"unparsable: {exc}")
        return
    defaults = limits.get("defaults") or {}
    if not defaults.get("rate_limit_per_minute"):
        fail("connector-limits", "defaults.rate_limit_per_minute missing")
        return
    for group in ("literature", "agricultural"):
        for name, cfg in (limits.get(group) or {}).items():
            rate = (cfg or {}).get("rate_limit_per_minute")
            if not isinstance(rate, (int, float)) or rate <= 0:
                fail("connector-limits", f"{name} rate_limit_per_minute must be positive")
                return
    ok("connector-limits")


def main() -> int:
    check_policy_integrity()
    check_agent_config()
    check_contracts()
    check_agents_md()
    check_secrets()
    check_verification_config()
    check_connector_limits()

    failed = False
    for check, outcome in _CHECKS:
        print(f"{outcome}  {check}")
        if outcome.startswith("FAIL"):
            failed = True
    print(f"\nUAASOP gate: {'FAILED' if failed else 'PASSED'} ({len(_CHECKS)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
