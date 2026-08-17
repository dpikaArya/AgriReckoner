"""Keep these tests out of the project's real artifact directories.

Several `src/ml` classes default their output to a working-directory-relative path
(`reports/`, `models/`, `features/`). Running the suite from the repository root therefore
wrote synthetic fixtures alongside genuine pipeline output, where nothing distinguished
them: `reports/leakage_detection_report.json` recorded `leakage_found: false` derived from
`make_regression` noise, and `models/ensemble_weights.json` carried `training_r2: 0.9928`
from the same fixtures. Both are the kind of artefact that gets quoted as a result.

Each test here runs in its own temporary directory, so those defaults land somewhere
harmless. Tests that need a real path should pass `output_dir` explicitly.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_working_directory(tmp_path, monkeypatch):
    """Run every test in this package from a scratch directory."""
    monkeypatch.chdir(tmp_path)
    yield
    assert os.getcwd() == str(tmp_path)
