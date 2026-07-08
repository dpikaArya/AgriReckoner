"""
Universal Agricultural Schema Generator (Merge Agent)
======================================================
BACKWARD-COMPATIBLE WRAPPER
Refactored into ADES (Agricultural Data Engineering System).

Legacy entry point: calls the ADES orchestrator pipeline
while preserving the original API for existing callers.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "master_datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline(input_path=None):
    """
    Legacy entry point.  Delegates to the ADES orchestrator.

    Parameters
    ----------
    input_path : str or Path, optional
        If None, auto-detect .xlsx files in data/master_datasets/.

    Returns
    -------
    pd.DataFrame : the final merged & engineered dataset.
    """
    from ades.config.settings import ADESSettings
    from ades.orchestrator import Orchestrator
    from ades.utils.logging_utils import setup_logging

    settings = ADESSettings()
    setup_logging(level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

    if input_path is None:
        xlsx_files = sorted(settings.DATA_DIR.glob("*.xlsx"))
        if xlsx_files:
            input_path = str(xlsx_files[0])
        else:
            log.info("No input file specified.  Looking for .xlsx in %s ...", DATA_DIR)
            import sys as _sys
            _sys.exit(1)

    orch = Orchestrator(settings=settings)
    df = orch.run(filepath=str(input_path))
    return df


if __name__ == "__main__":
    try:
        path = sys.argv[1] if len(sys.argv) > 1 else None
        df = run_pipeline(path)
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        sys.exit(1)
