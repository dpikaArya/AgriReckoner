"""Standalone runner for the Literature Intelligence Module.

Usage:
    python run_literature.py [--terms "field experiment rice"] [--full]
                             [--skip-agri] [--dry-run]
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("Literature")

from agri_ai_agent.literature.config import LiteratureConfig  # noqa: E402
from agri_ai_agent.literature.pipeline import LiteraturePipeline  # noqa: E402

DEFAULT_TERMS = (
    "wheat field experiment randomized complete block design",
    "rice split plot factorial fertilizer nitrogen phosphorus potassium",
    "maize irrigation water use efficiency cultivar yield",
    "wheat sowing date plant density micronutrient zinc boron yield",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="AAIF Literature Intelligence Module")
    parser.add_argument("--terms", "-t", nargs="+", default=list(DEFAULT_TERMS), help="Boolean search terms")
    parser.add_argument("--full", action="store_true", help="Re-process all cached records")
    parser.add_argument("--skip-agri", action="store_true", help="Skip agricultural data connectors")
    parser.add_argument("--max-verify-pdfs", type=int, default=200, help="Cap for PDF HEAD verifications")
    parser.add_argument("--dry-run", action="store_true", help="Print connector plan and exit")
    parser.add_argument("--connector", help="Run only one literature connector (source_name)")
    args = parser.parse_args()

    config = LiteratureConfig.from_env(root=BASE_DIR)
    config.search_terms = tuple(args.terms)
    if args.skip_agri:
        config.search_terms = config.search_terms  # agri handled in pipeline separately

    pipeline = LiteraturePipeline(config=config)

    if args.connector:
        from agri_ai_agent.literature.registry import discover_literature_connectors

        wanted = {c.source_name: c for c in discover_literature_connectors()}
        if args.connector not in wanted:
            log.error("Unknown connector: %s (available: %s)", args.connector, ", ".join(wanted))
            return 2
        pipeline._connector_classes = [wanted[args.connector]]
        pipeline.connectors = []
        pipeline._lit_by_source = {}
        pipeline._build_literature_connectors()

    if args.dry_run:
        log.info(
            "Dry run: literature=%s",
            ", ".join(c.source_name for c in pipeline.enabled_connectors),
        )
        log.info(
            "Dry run: agri=%s",
            ", ".join(c.source_name for c in pipeline.agri_manager.enabled_connectors),
        )
        return 0

    report = pipeline.run(
        terms=config.search_terms,
        full=args.full,
        max_verify_pdfs=args.max_verify_pdfs,
        skip_agri=args.skip_agri,
    )
    log.info("\n%s", report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
