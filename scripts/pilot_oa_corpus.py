"""Measure whether open-access papers can supply treatment-level training rows.

The corpus question is not "are there papers" — Europe PMC has ~16,800 open-access
fertiliser field-trial papers with full-text XML. It is whether a paper yields rows a model
can learn from, which needs two things that are usually reported in different places:

  * a yield per treatment, which lives in a table, and
  * the fertiliser dose each treatment received, which is usually a legend in the methods
    ("T1 = 0 kg N/ha, T2 = 60 kg N/ha") keyed to codes used in that table.

A yield with no dose attached is not a training row, so this script reports the two rates
separately. Numbers are parsed from the XML by code; the model is asked only to interpret
headers and the dose legend, never to read or repeat a value.

Usage:
    python scripts/pilot_oa_corpus.py --papers 10 --out pilot_results.json
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
TIMEOUT = 90
QUERY = (
    '(fertilizer OR fertiliser) AND yield AND ("field experiment" OR "field trial") '
    "AND OPEN_ACCESS:Y AND HAS_FT:Y"
)
# Papers separate mass and area with a slash, a space, a middle dot or nothing at all,
# and use either a hyphen or a Unicode minus in the exponent.
YIELD_UNITS = re.compile(r"\b(kg|t|q|g|mg)[\s/·.]*(ha|hm|m)\b", re.I)
TREATMENT_CODE = re.compile(
    r"^\s*(T\s?-?\d+|N\s?\d+|F\s?\d+|M\s?\d+|CK|RDF|control|check|absolute)\b", re.I
)
# Many papers put the rate inside the treatment label itself ("N2 (300 kg/ha)"), so the
# dose is recoverable without any methods legend at all.
INLINE_DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|t|q)[\s/·.]*(?:ha|hm)", re.I)
SUMMARY_ROW = re.compile(r"^\s*(range|mean|sd|s\.d|cv|se|sem|c\.?d\.?|lsd|total|average)\b", re.I)


@dataclass
class PaperResult:
    """What one paper contributed, and where it fell short."""

    pmcid: str
    licence: str = ""
    yield_tables: int = 0
    treatment_rows: int = 0
    has_yield_unit: bool = False
    has_replication: bool = False
    has_dispersion: bool = False
    uses_codes: bool = False
    legend_found: bool = False
    doses_joined: int = 0
    eligible: bool = False
    notes: list = field(default_factory=list)


def fetch(url):
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read().decode("utf-8", "replace")


def search(limit, query=QUERY):
    url = f"{EPMC}/search?query={urllib.parse.quote(query)}&format=json&pageSize={limit}&resultType=core"
    data = json.loads(fetch(url))
    return [
        (r["pmcid"], r.get("license", "?")) for r in data["resultList"]["result"] if r.get("pmcid")
    ]


def _cell_text(node):
    text = "".join(node.itertext()).replace("\u2212", "-").replace("\u2009", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_table(table_el):
    """Return (caption, header_rows, body_rows) with rowspan/colspan expanded.

    Spans are near-universal in these tables; ignoring them shifts every value into the
    wrong column, which would silently corrupt any number read downstream.
    """
    caption = ""
    for tag in ("caption", "label", "title"):
        node = table_el.find(f".//{tag}")
        if node is not None:
            caption += " " + _cell_text(node)
    grid, pending = [], {}
    for row in table_el.iter("tr"):
        out, col = [], 0
        while col in pending:
            text, left = pending[col]
            out.append(text)
            pending[col] = (text, left - 1) if left > 1 else None
            if pending[col] is None:
                del pending[col]
            col += 1
        for cell in row:
            if cell.tag not in ("td", "th"):
                continue
            text = _cell_text(cell)
            span = int(cell.get("colspan", 1) or 1)
            down = int(cell.get("rowspan", 1) or 1)
            for _ in range(span):
                out.append(text)
                if down > 1:
                    pending[col] = (text, down - 1)
                col += 1
                while col in pending:
                    held, left = pending[col]
                    out.append(held)
                    pending[col] = (held, left - 1) if left > 1 else None
                    if pending[col] is None:
                        del pending[col]
                    col += 1
        if out:
            grid.append(out)
    if not grid:
        return caption.strip(), [], []
    return caption.strip(), grid[:1], grid[1:]


def yield_tables(root):
    """Tables whose caption or header mentions yield."""
    found = []
    for table in root.iter("table-wrap"):
        caption, header, body = parse_table(table)
        blob = caption + " " + " ".join(header[0] if header else [])
        if re.search(r"yield", blob, re.I):
            found.append((caption, header, body))
    return found


def treatment_column(header, body):
    """Index of the column holding treatment labels.

    It is not reliably the first column: tables are commonly keyed by year or site first,
    so assuming position silently reads the wrong field for every row.
    """
    head = header[0] if header else []
    for index, name in enumerate(head):
        if re.search(r"\btreatment|\btreatments\b|\bfertili[sz]er\b", name, re.I):
            return index
    best, best_hits = 0, 0
    for index in range(max((len(r) for r in body), default=1)):
        hits = sum(1 for row in body if len(row) > index and TREATMENT_CODE.match(row[index]))
        if hits > best_hits:
            best, best_hits = index, hits
    return best


def treatment_rows(header, body, column=None):
    """Rows that look like a treatment with at least one numeric value."""
    column = treatment_column(header, body) if column is None else column
    rows = []
    for row in body:
        if len(row) <= column:
            continue
        label = row[column]
        if not label or SUMMARY_ROW.match(label):
            continue
        if not TREATMENT_CODE.match(label) and not re.search(r"\d", label):
            continue
        others = [c for i, c in enumerate(row) if i != column]
        if sum(bool(re.fullmatch(r"[\d.]+", c.split("±")[0].strip())) for c in others) >= 1:
            rows.append(row)
    return rows


def methods_text(root, limit=18000):
    """The methods/materials section, where the dose legend usually sits."""
    chunks = []
    for sec in root.iter("sec"):
        title = sec.find("title")
        if title is not None and re.search(r"method|material", _cell_text(title), re.I):
            chunks.append(" ".join(sec.itertext()))
    if not chunks:
        chunks = [" ".join(root.itertext())[:limit]]
    return re.sub(r"\s+", " ", " ".join(chunks))[:limit]


def ask(messages, schema, api_key):
    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "r", "strict": True, "schema": schema},
        },
    }
    request = urllib.request.Request(
        OPENROUTER,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        body = json.load(response)
    return json.loads(body["choices"][0]["message"]["content"])


LEGEND_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "legend_present": {"type": "boolean"},
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "code": {"type": "string"},
                    "n_kg_ha": {"type": "number"},
                    "p_kg_ha": {"type": "number"},
                    "k_kg_ha": {"type": "number"},
                    "source_quote": {"type": "string"},
                },
                "required": ["code", "n_kg_ha", "p_kg_ha", "k_kg_ha", "source_quote"],
            },
        },
        "replications": {"type": "integer"},
    },
    "required": ["legend_present", "entries", "replications"],
}

LEGEND_SYSTEM = (
    "You read the methods section of an agronomy field trial. Report the fertiliser dose "
    "assigned to each treatment code exactly as written, with the sentence it came from in "
    "source_quote. Use -1 for any dose the text does not state. Never infer or complete a "
    "dose that is not written down. If no code-to-dose legend exists, set legend_present false "
    "and return no entries."
)


def screen_paper(pmcid, licence, api_key):
    result = PaperResult(pmcid=pmcid, licence=licence)
    try:
        root = ET.fromstring(fetch(f"{EPMC}/{pmcid}/fullTextXML"))
    except Exception as exc:
        result.notes.append(f"xml unavailable: {exc}")
        return result

    tables = yield_tables(root)
    result.yield_tables = len(tables)
    if not tables:
        result.notes.append("no yield table")
        return result

    # Rank by "carries a yield unit" first, then by size. Picking purely by row count
    # selects recommendation lookup tables over the experiment's own results table.
    best, best_caption, best_header, best_col, best_rank = [], "", [], 0, (0, 0)
    for caption, header, body in tables:
        column = treatment_column(header, body)
        rows = treatment_rows(header, body, column)
        blob = " ".join(header[0] if header else []) + " " + caption
        rank = (1 if YIELD_UNITS.search(blob) else 0, len(rows))
        if rows and rank > best_rank:
            best, best_caption, best_header, best_col, best_rank = (
                rows,
                caption,
                header,
                column,
                rank,
            )
    result.treatment_rows = len(best)
    if len(best) < 2:
        result.notes.append("no table with >=2 treatment rows")
        return result

    header_blob = " ".join(best_header[0] if best_header else []) + " " + best_caption
    result.has_yield_unit = bool(YIELD_UNITS.search(header_blob))
    # Dispersion is reported three ways: inline "mean +- sd", a CD/LSD/SEm row beneath the
    # treatments, or a compact-letter-display suffix. The CD row is filtered out of the
    # treatment rows, so it has to be looked for in the unfiltered table.
    all_rows = [r for _, _, body in tables for r in body]
    result.has_dispersion = (
        any("±" in c for row in best for c in row)
        or any(
            SUMMARY_ROW.match(r[0]) and re.search(r"\b(cd|lsd|sem|se)\b", r[0], re.I)
            for r in all_rows
            if r
        )
        or bool(re.search(r"\b(cd|lsd|sem)\b", header_blob, re.I))
        or any(re.search(r"\d\s*[a-e]{1,2}$", c.strip()) for row in best for c in row)
    )
    result.uses_codes = any(TREATMENT_CODE.match(row[best_col]) for row in best)

    legend = {"legend_present": False, "entries": [], "replications": 0}
    try:
        legend = ask(
            [
                {"role": "system", "content": LEGEND_SYSTEM},
                {"role": "user", "content": f"Methods of {pmcid}:\n\n{methods_text(root)}"},
            ],
            LEGEND_SCHEMA,
            api_key,
        )
    except Exception as exc:
        result.notes.append(f"legend call failed: {exc}")

    result.has_replication = int(legend.get("replications") or 0) >= 2
    result.legend_found = bool(legend.get("legend_present")) and bool(legend.get("entries"))

    header_cells = best_header[0] if best_header else []
    dose_cols = [
        i
        for i, name in enumerate(header_cells)
        if re.search(r"nutrient|applied|added|dose|rate", name, re.I) and YIELD_UNITS.search(name)
    ]
    inline = sum(1 for row in best if INLINE_DOSE.search(row[best_col]))
    if not inline and dose_cols:
        inline = sum(
            1
            for row in best
            if any(re.fullmatch(r"[\d.]+", row[i].strip()) for i in dose_cols if i < len(row))
        )
    if inline >= 2:
        result.doses_joined = inline
        result.notes.append(f"dose inline in treatment label ({inline} rows)")

    if result.legend_found and result.doses_joined < 2:
        codes = {
            re.sub(r"[^a-z0-9]", "", e["code"].lower()): e
            for e in legend["entries"]
            if any(e.get(k, -1) >= 0 for k in ("n_kg_ha", "p_kg_ha", "k_kg_ha"))
        }
        for row in best:
            if codes.get(re.sub(r"[^a-z0-9]", "", row[best_col].lower())):
                result.doses_joined += 1

    result.eligible = (
        result.treatment_rows >= 2
        and result.has_yield_unit
        and result.has_dispersion
        and result.doses_joined >= 2
    )
    if not result.eligible:
        missing = [
            name
            for name, ok in [
                ("yield unit", result.has_yield_unit),
                ("dispersion", result.has_dispersion),
                ("dose join", result.doses_joined >= 2),
            ]
            if not ok
        ]
        result.notes.append("missing: " + ", ".join(missing))
    return result


def report(results):
    n = len(results)
    eligible = [r for r in results if r.eligible]
    coded = [r for r in results if r.uses_codes]
    joined = [r for r in coded if r.doses_joined >= 2]
    rows = sum(r.doses_joined for r in eligible)

    print(f"\n{'=' * 70}\nPILOT-1 — {n} open-access papers\n{'=' * 70}")
    for r in results:
        flag = "ELIGIBLE" if r.eligible else "no      "
        print(
            f"  {flag} {r.pmcid:12s} tables={r.yield_tables} rows={r.treatment_rows:2d} "
            f"unit={'Y' if r.has_yield_unit else 'n'} disp={'Y' if r.has_dispersion else 'n'} "
            f"codes={'Y' if r.uses_codes else 'n'} legend={'Y' if r.legend_found else 'n'} "
            f"joined={r.doses_joined:2d}  {'; '.join(r.notes)[:44]}"
        )
    print(f"\n  eligibility rate : {len(eligible)}/{n} ({100 * len(eligible) / max(n, 1):.0f}%)")
    if coded:
        print(
            f"  dose-join rate   : {len(joined)}/{len(coded)} "
            f"({100 * len(joined) / len(coded):.0f}%) of code-using papers"
        )
    else:
        print("  dose-join rate   : no paper used treatment codes")
    print(
        f"  usable rows      : {rows} (mean {rows / max(len(eligible), 1):.1f} per eligible paper)"
    )
    print("\n  STOP if eligibility < 15% or dose-join < 50%; PASS at >= 25% and >= 70%.")
    return {
        "papers": n,
        "eligible": len(eligible),
        "coded": len(coded),
        "joined": len(joined),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=10)
    parser.add_argument("--query", default=QUERY, help="Europe PMC query")
    parser.add_argument("--out", default="pilot_results.json")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    results = []
    for pmcid, licence in search(args.papers, args.query):
        print(f"  screening {pmcid} ...", flush=True)
        results.append(screen_paper(pmcid, licence, api_key))

    summary = report(results)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "papers": [vars(r) for r in results]}, handle, indent=2)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
