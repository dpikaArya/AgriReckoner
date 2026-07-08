"""
ADES Command Line Interface.

Usage:
    ades run input_dataset.xlsx
    ades learn --papers ./new_papers/
    ades --version
"""

import sys
import argparse
from pathlib import Path

from ades import __version__
from ades.config.settings import ADESSettings
from ades.orchestrator import Orchestrator
from ades.agents.continuous_learning_agent import ContinuousLearningAgent
from ades.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        prog="ades",
        description="Agricultural Data Engineering System (ADES) - "
                    "Agentic AI pipeline for agricultural ML datasets",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    run_parser = subparsers.add_parser("run", help="Run the full ADES pipeline")
    run_parser.add_argument("input", nargs="?", type=str, help="Input dataset path")
    run_parser.add_argument("--input", "-i", dest="input_opt", type=str, help="Input dataset path")
    run_parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    run_parser.add_argument("--log-level", type=str, default=None, help="Log level (DEBUG, INFO, WARNING)")
    run_parser.add_argument("--incremental", action="store_true", help="Enable incremental mode")
    run_parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")

    learn_parser = subparsers.add_parser("learn", help="Run continuous learning cycle from new papers")
    learn_parser.add_argument("--papers", "-p", type=str, required=True, help="Directory containing new research papers (PDFs)")
    learn_parser.add_argument("--master", "-m", type=str, default=None, help="Existing master dataset path (CSV/XLSX)")
    learn_parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
    learn_parser.add_argument("--force", action="store_true", help="Force retrain even without new papers")

    args = parser.parse_args()

    if args.version:
        print(f"ADES v{__version__}")
        return

    settings = ADESSettings()

    if args.command == "run":
        input_path = args.input or args.input_opt
        if not input_path:
            xlsx_files = sorted(settings.DATA_DIR.glob("*.xlsx"))
            if xlsx_files:
                input_path = str(xlsx_files[0])
                print(f"Auto-detected input: {input_path}")
            else:
                print("Error: No input file provided and no .xlsx found in data/master_datasets/")
                sys.exit(1)

        if args.output:
            settings.OUTPUT_DIR = Path(args.output)
        if args.log_level:
            settings.LOG_LEVEL = args.log_level
        if args.incremental:
            settings.INCREMENTAL_MODE = True
        if args.no_checkpoint:
            settings.CHECKPOINT_ENABLED = False

        setup_logging(level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

        orch = Orchestrator(settings=settings)
        try:
            orch.run(filepath=input_path)
        except Exception as e:
            print(f"Pipeline failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "learn":
        if args.output:
            settings.OUTPUT_DIR = Path(args.output)
        setup_logging(level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

        master_df = None
        if args.master:
            mpath = Path(args.master)
            if mpath.suffix == ".csv":
                import pandas as pd
                master_df = pd.read_csv(mpath)
                print(f"Loaded master dataset: {mpath.name} ({len(master_df)} rows)")
            elif mpath.suffix in (".xlsx", ".xls"):
                import pandas as pd
                master_df = pd.read_excel(mpath)
                print(f"Loaded master dataset: {mpath.name} ({len(master_df)} rows)")

        agent = ContinuousLearningAgent(settings=settings)
        import pandas as pd
        result_df = agent.run(
            df=master_df if master_df is not None else pd.DataFrame(),
            papers_dir=args.papers,
            force_retrain=args.force,
        )
        print(f"Continuous learning cycle complete. Dataset: {len(result_df)} rows.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
