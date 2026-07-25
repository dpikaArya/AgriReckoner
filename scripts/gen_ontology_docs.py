"""Regenerate a Markdown ontology table from spec/uams_ontology.yaml.

Writes to stdout by default, or to a path via --out. This lets a human-readable
registry (like spec/Ontology_Registry.md) be regenerated from the machine-
readable source of truth. It never overwrites spec/Ontology_Registry.md unless
that exact path is passed explicitly to --out.

Usage:
    python scripts/gen_ontology_docs.py                 # print to stdout
    python scripts/gen_ontology_docs.py --out doc.md    # write to a file
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agri_ai_agent.ontology.registry import (  # noqa: E402
    DEFAULT_REGISTRY_PATH,
    load_registry,
)

_TABLE_HEADER = (
    "| Variable | Group | Terms (CURIE) | Unit (UCUM) | Range | Synonyms |\n"
    "|----------|-------|---------------|-------------|-------|----------|\n"
)
_MAX_SYNONYMS = 6


def _fmt_terms(terms: list[dict]) -> str:
    """Render a column's term CURIEs as a cell, or a dash when empty."""
    if not terms:
        return "—"
    return "; ".join(term["curie"] for term in terms)


def _fmt_range(validation: dict) -> str:
    """Render the (min, max) validation range, or a dash when unset."""
    low, high = validation.get("min"), validation.get("max")
    if low is None and high is None:
        return "—"
    return f"{low}–{high}"


def _fmt_synonyms(synonyms: list[str]) -> str:
    """Render up to _MAX_SYNONYMS synonyms, eliding the rest."""
    if not synonyms:
        return "—"
    shown = synonyms[:_MAX_SYNONYMS]
    suffix = " …" if len(synonyms) > _MAX_SYNONYMS else ""
    return ", ".join(shown) + suffix


def _row(column: str, block: dict) -> str:
    """Render one Markdown table row for a column."""
    return (
        f"| {column} | {block['group']} | {_fmt_terms(block['terms'])} "
        f"| {block.get('unit_ucum') or '—'} | {_fmt_range(block['validation'])} "
        f"| {_fmt_synonyms(block['synonyms'])} |"
    )


def render_markdown(registry: dict) -> str:
    """Return the full Markdown document as a string."""
    prov = registry.get("provenance", {})
    lines = [
        "# UAMS Ontology Registry (generated)",
        "",
        f"**Version:** {registry.get('version', 'unknown')}  ",
        f"**Namespace:** {registry.get('namespace', 'unknown')}  ",
        f"**Provenance note:** {prov.get('generated_note', '—')}",
        "",
        "Generated from `spec/uams_ontology.yaml` by "
        "`scripts/gen_ontology_docs.py`. Do not edit by hand.",
        "",
        _TABLE_HEADER.rstrip("\n"),
    ]
    lines.extend(_row(col, block) for col, block in registry["columns"].items())
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH),
                        help="Path to the ontology YAML.")
    parser.add_argument("--out", default=None,
                        help="Output path; prints to stdout when omitted.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Render the Markdown doc and write it to --out or stdout."""
    args = _parse_args(argv)
    markdown = render_markdown(load_registry(args.registry))
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(markdown)
    return 0


def _smoke() -> None:
    """Side-effect-free self-check: render and assert the table is complete."""
    doc = render_markdown(load_registry(str(DEFAULT_REGISTRY_PATH)))
    assert doc.count("\n|") >= 138, "expected >=138 table rows"
    assert "Soil_pH" in doc and "AGROVOC:c_5188" in doc
    print("gen_ontology_docs smoke OK", file=sys.stderr)


if __name__ == "__main__":
    _smoke()
    sys.exit(main(sys.argv[1:]))
