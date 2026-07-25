"""
AgriAI CLI: run pipeline, continuous learning, and utility commands.
"""

import argparse
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

    extract_parser = sub.add_parser("extract", help="Extract UAMS rows from PDFs via grounded LLM extraction")
    extract_parser.add_argument("--papers", "-p", required=True, help="Directory of PDF papers")
    extract_parser.add_argument("--out", "-o", help="Output CSV path (default: OUTPUT_DIR/LLM_Extracted_Schema.csv)")

    sub.add_parser("version", help="Print version info")

    args = parser.parse_args()

    if args.command == "version":
        from agri_ai_agent import __version__
        print(f"AgriAI v{__version__}")
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

    if args.command == "extract":
        _run_extract(settings, args)
        return

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


def _run_extract(settings, args):
    """Extract grounded UAMS rows from a directory of PDFs via the LLM extractor."""
    from agri_ai_agent.agents.llm_extraction_agent import LLMExtractionAgent
    from agri_ai_agent.extractors.pdf_reader import read_papers_dir

    papers = read_papers_dir(args.papers)
    if not papers:
        logger.warning("No readable PDFs found in %s", args.papers)
        return

    agent = LLMExtractionAgent(settings=settings)
    contract = agent.run(df=None, papers=papers)
    if contract.warnings:
        logger.warning("%s", "; ".join(contract.warnings))

    df = agent.dataframe
    if df is None or df.empty:
        logger.info("No rows extracted (no OpenAI key, or nothing grounded).")
        return

    out_path = Path(args.out) if args.out else settings.OUTPUT_DIR / "LLM_Extracted_Schema.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Extracted %d papers -> %s", len(papers), out_path)


if __name__ == "__main__":
    main()
