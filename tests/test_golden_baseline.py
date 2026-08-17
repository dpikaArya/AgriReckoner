"""Golden-baseline guard.

The committed outputs/ and models/ were produced by the validated legacy monolith and are
frozen at git tag ``legacy-monolith-v1``. This test fails if any of them is modified, so the
agent pipeline (or anything else) cannot silently overwrite the validated reference before a
replacement is proven to reproduce it.
"""

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "baseline" / "legacy_monolith_v1.sha256.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="baseline manifest absent")
def test_committed_artifacts_match_frozen_checksums():
    manifest = json.loads(MANIFEST.read_text())
    changed = []
    for rel_path, meta in manifest["files"].items():
        path = REPO / rel_path
        if not path.exists():
            changed.append(f"missing:{rel_path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            changed.append(f"modified:{rel_path}")
    assert not changed, f"frozen baseline artifacts changed: {changed[:5]}"


def test_committed_schema_matches_uams_columns():
    schema_csv = REPO / "outputs" / "Universal_Agricultural_Schema.csv"
    if not schema_csv.exists():
        pytest.skip("committed schema CSV absent")
    from agri_ai_agent.config.schema import UAMS_COLUMNS

    header = schema_csv.read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert len(header) == len(UAMS_COLUMNS) == 296


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
