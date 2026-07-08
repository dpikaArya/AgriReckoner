"""
AgriAI CLI: run pipeline, continuous learning, and utility commands.
"""

import argparse
import sys
from pathlib import Path

from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.orchestrator import Orchestrator
from agri_ai_agent.utils.logging_utils import get_logger

logger = get_logger("CLI")


def main():
    parser = argparse.ArgumentParser(
        prog="agriai",
        description="AgriAI — Self-improving agricultural ML pipeline",
    )
    parser.add_argument("--settings", help="Path to custom settings JSON")

    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--file", "-f", help="Input data file (CSV/Excel/Parquet)")
    run_parser.add_argument("--papers", "-p", help="Directory of research papers (PDFs)")

    learn_parser = sub.add_parser("learn", help="Run continuous learning cycle")
    learn_parser.add_argument("--papers", "-p", required=True, help="Directory with new PDF papers")
    learn_parser.add_argument("--force-retrain", action="store_true", help="Force retrain even without improvement")
    learn_parser.add_argument("--update-fuzzy-rules", action="store_true", help="Update fuzzy rules based on data drift")

    sub.add_parser("version", help="Print version info")

    args = parser.parse_args()

    if args.command == "version":
        print("AgriAI v1.0.0")
        return

    if args.command is None:
        parser.print_help()
        return

    settings = AgriAISettings()
    if args.settings:
        import json
        with open(args.settings) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if hasattr(settings, k.upper()):
                setattr(settings, k.upper(), v)

    settings.ensure_dirs()

    orchestrator = Orchestrator(settings=settings)

    if args.command == "run":
        df = orchestrator.run(filepath=args.file, papers_dir=args.papers)
        logger.info("Pipeline finished. Output rows: %d", len(df))

    elif args.command == "learn":
        orchestrator.run_continuous(
            papers_dir=args.papers,
            force_retrain=args.force_retrain,
            update_fuzzy_rules=args.update_fuzzy_rules,
        )
        logger.info("Continuous learning cycle complete.")


if __name__ == "__main__":
    main()
