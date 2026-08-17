"""The leakage check must be able to report leakage.

Regression: the stage looked for a two-column report layout that the producer has never
written, so it found nothing every time and published "Leakage detected: 0" while the guard
was flagging 38 features. A check that cannot fail is worse than no check, because it is
quoted as evidence the pipeline is clean.
"""

import pandas as pd
import pytest

from evaluation.stage08_leakage import _flagged_features


def test_the_layout_the_pipeline_actually_writes_is_read():
    report = pd.DataFrame({"leaked_feature": ["Yield_per_Plot", "Harvest_Index", "Protein"]})
    assert _flagged_features(report) == ["Yield_per_Plot", "Harvest_Index", "Protein"]


def test_a_utf8_bom_in_the_header_does_not_hide_the_column():
    """The report is written with a BOM, which renames the column if not stripped."""
    report = pd.DataFrame({"﻿leaked_feature": ["Yield_per_Plot"]})
    assert _flagged_features(report) == ["Yield_per_Plot"]


def test_the_legacy_two_column_layout_still_works():
    report = pd.DataFrame(
        {"Feature": ["Soil_pH", "Harvest_Index"], "Available_Before_Prediction": [True, False]}
    )
    assert _flagged_features(report) == ["Harvest_Index"]


@pytest.mark.parametrize("frame", [None, pd.DataFrame()])
def test_no_report_means_nothing_flagged(frame):
    assert _flagged_features(frame) == []


def test_an_unrecognised_layout_raises_instead_of_reporting_all_clear():
    """Silence here is what let the broken contract survive; it must be loud."""
    with pytest.raises(ValueError, match="unrecognised layout"):
        _flagged_features(pd.DataFrame({"something_else": [1, 2]}))


def test_the_real_committed_report_is_readable():
    from pathlib import Path

    path = Path(__file__).parent.parent / "outputs" / "Leakage_Report.csv"
    if not path.exists():
        pytest.skip("no committed leakage report")
    flagged = _flagged_features(pd.read_csv(path, encoding="utf-8-sig"))
    assert len(flagged) > 0, "the committed report lists leaked features; the stage must see them"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
