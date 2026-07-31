"""Build treatment-level training rows from open-access agronomy papers.

Division of labour, which the pilot established the hard way: rules can parse a table but
cannot decide which table is the experiment. Ranking by size picks recommendation lookup
tables; keying on the first column reads the year instead of the treatment. So the model is
asked only to *choose* — which table, which column is the treatment, which is the yield,
what unit it is in — and every number is then read out of the XML by code. The model never
sees a value it could restate, so no reported figure can be an invention.

Each emitted row carries the table and row it came from and the verbatim source cell, so any
value can be checked against the paper without rerunning anything.

Usage:
    python scripts/extract_oa_corpus.py --papers 100 --out corpus.json
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
PRICE_IN, PRICE_OUT = 0.048e-6, 0.193e-6  # USD per token, for the running cost report
TIMEOUT = 120
DEFAULT_QUERY = (
    '"grain yield" AND (fertilizer OR fertiliser) AND ("field experiment" OR "field trial") '
    "AND OPEN_ACCESS:Y AND HAS_FT:Y"
)
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")

SELECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "table_index": {"type": "integer"},
        "is_experimental_results": {"type": "boolean"},
        "treatment_column": {"type": "integer"},
        "yield_column": {"type": "integer"},
        "yield_unit": {"type": "string"},
        "dose_columns": {"type": "array", "items": {"type": "integer"}},
        "dispersion_present": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "table_index",
        "is_experimental_results",
        "treatment_column",
        "yield_column",
        "yield_unit",
        "dose_columns",
        "dispersion_present",
        "reason",
    ],
}

SELECT_SYSTEM = """You are shown the caption, header and first rows of every table in an
agronomy paper. Choose the ONE table that reports a crop yield for each experimental
treatment, and identify its columns by zero-based index.

A table qualifies only if its rows are the trial's own treatments and one column is a yield
per unit area. These do NOT qualify:
  - summary statistics (Range / Mean +- SD / CV / ANOVA / correlation matrices)
  - regression or response-surface equations
  - recommendation or ready-reckoner lookup tables keyed by soil test value
  - yield COMPONENTS only (spikelets, 1000-grain weight, seed setting rate) with no yield
  - literature review tables citing other studies

Column indices refer to the header you are shown. The treatment column is often not the
first: tables are commonly keyed by year or site first. dose_columns are any columns giving
the fertiliser rate applied (e.g. "Nutrients added (kg ha-1)", "N (kg/ha)").

If no table qualifies, set is_experimental_results false and table_index -1. Do not guess.
Never report a numeric value; you are choosing columns, not reading data."""


class Ledger:
    """Running token and cost tally, so a run can be stopped before a budget is passed."""

    def __init__(self, budget_usd):
        self.budget = budget_usd
        self.tokens_in = self.tokens_out = 0
        self.calls = 0

    def add(self, usage):
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        self.calls += 1

    @property
    def cost(self):
        return self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT

    def exhausted(self):
        return self.cost >= self.budget

    def __str__(self):
        return (
            f"{self.calls} calls, {self.tokens_in:,} in / {self.tokens_out:,} out, "
            f"${self.cost:.4f} of ${self.budget:.2f}"
        )


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                return response.read().decode("utf-8", "replace")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def search(limit, query):
    """Page through Europe PMC for open-access papers with full text."""
    out, cursor = [], "*"
    while len(out) < limit:
        url = (
            f"{EPMC}/search?query={urllib.parse.quote(query)}&format=json"
            f"&pageSize=100&cursorMark={urllib.parse.quote(cursor)}&resultType=core"
        )
        data = json.loads(fetch(url))
        results = data["resultList"]["result"]
        if not results:
            break
        out += [(r["pmcid"], r.get("license", "?")) for r in results if r.get("pmcid")]
        nxt = data.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
    return out[:limit]


def cell_text(node):
    text = "".join(node.itertext()).replace("−", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_table(table_el):
    """Return (caption, rows) with rowspan/colspan expanded into a rectangular grid."""
    caption = " ".join(
        cell_text(node) for tag in ("label", "caption") for node in table_el.findall(f".//{tag}")
    )
    grid, pending = [], {}
    for row in table_el.iter("tr"):
        out, col = [], 0
        while col in pending:
            text, left = pending.pop(col)
            out.append(text)
            if left > 1:
                pending[col] = (text, left - 1)
            col += 1
        for cell in row:
            if cell.tag not in ("td", "th"):
                continue
            text = cell_text(cell)
            for _ in range(int(cell.get("colspan", 1) or 1)):
                out.append(text)
                down = int(cell.get("rowspan", 1) or 1)
                if down > 1:
                    pending[col] = (text, down - 1)
                col += 1
                while col in pending:
                    held, left = pending.pop(col)
                    out.append(held)
                    if left > 1:
                        pending[col] = (held, left - 1)
                    col += 1
        if out:
            grid.append(out)
    return caption.strip(), grid


def tables_of(root):
    return [parse_table(t) for t in root.iter("table-wrap")]


def summarise(tables, max_rows=4, width=34):
    """Compact view of every table: enough to choose one, too little to read data from."""
    lines = []
    for index, (caption, grid) in enumerate(tables):
        if not grid:
            continue
        lines.append(f"[table {index}] {caption[:160]}")
        for row in grid[:max_rows]:
            cells = " | ".join(f"{i}:{c[:width]}" for i, c in enumerate(row[:9]))
            lines.append(f"    {cells}")
        lines.append(f"    ({len(grid)} rows)")
    return "\n".join(lines)


def ask(messages, schema, api_key, ledger):
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
    ledger.add(body.get("usage") or {})
    return json.loads(body["choices"][0]["message"]["content"])


def number_in(text):
    """First numeric value in a cell, ignoring dispersion and significance letters."""
    head = re.split(r"[±±]", str(text))[0].strip()
    head = re.sub(r"[a-zA-Z\s*]+$", "", head).strip()
    return float(head) if NUMERIC.match(head) else None


def rows_from(grid, choice):
    """Read treatment/yield/dose values out of the chosen columns, by code."""
    treat_col = choice["treatment_column"]
    yield_col = choice["yield_column"]
    dose_cols = [c for c in choice.get("dose_columns") or [] if c >= 0]
    out = []
    for row_index, row in enumerate(grid):
        if max(treat_col, yield_col, *(dose_cols or [0])) >= len(row):
            continue
        label = row[treat_col].strip()
        value = number_in(row[yield_col])
        if not label or value is None:
            continue
        if re.match(r"^\s*(mean|range|cv|sd|se|sem|c\.?d\.?|lsd|total|average)\b", label, re.I):
            continue
        out.append(
            {
                "treatment": label,
                "yield_value": value,
                "yield_unit": choice.get("yield_unit", ""),
                "doses": [number_in(row[c]) for c in dose_cols],
                "row_index": row_index,
                "source_cell": row[yield_col],
            }
        )
    return out


def process(pmcid, licence, api_key, ledger):
    record = {"pmcid": pmcid, "licence": licence, "rows": [], "reason": ""}
    try:
        root = ET.fromstring(fetch(f"{EPMC}/{pmcid}/fullTextXML"))
    except Exception as exc:
        record["reason"] = f"xml unavailable: {exc}"
        return record

    tables = tables_of(root)
    if not tables:
        record["reason"] = "no tables"
        return record

    try:
        choice = ask(
            [
                {"role": "system", "content": SELECT_SYSTEM},
                {"role": "user", "content": f"Paper {pmcid}\n\n{summarise(tables)[:24000]}"},
            ],
            SELECT_SCHEMA,
            api_key,
            ledger,
        )
    except Exception as exc:
        record["reason"] = f"selection failed: {exc}"
        return record

    record["choice"] = choice
    if not choice["is_experimental_results"] or not 0 <= choice["table_index"] < len(tables):
        record["reason"] = choice.get("reason", "no qualifying table")[:160]
        return record

    caption, grid = tables[choice["table_index"]]
    record["caption"] = caption[:200]
    record["rows"] = rows_from(grid, choice)
    if not record["rows"]:
        record["reason"] = "chosen table yielded no readable rows"
    return record


def report(records, ledger):
    usable = [r for r in records if len(r["rows"]) >= 2]
    rows = sum(len(r["rows"]) for r in usable)
    with_dose = sum(
        1 for r in usable for row in r["rows"] if any(d is not None for d in row["doses"])
    )
    print(f"\n{'=' * 72}\nOPEN-ACCESS CORPUS EXTRACTION — {len(records)} papers\n{'=' * 72}")
    for r in records:
        mark = "OK " if len(r["rows"]) >= 2 else "-- "
        print(f"  {mark}{r['pmcid']:12s} rows={len(r['rows']):3d}  {r.get('reason', '')[:52]}")
    print(
        f"\n  papers with >=2 treatment rows : {len(usable)}/{len(records)} "
        f"({100 * len(usable) / max(len(records), 1):.0f}%)"
    )
    print(f"  treatment rows extracted       : {rows}")
    print(f"  rows carrying a dose           : {with_dose}")
    print(f"  mean rows per usable paper     : {rows / max(len(usable), 1):.1f}")
    print(
        f"  licences                       : { {r['licence'] for r in usable} if usable else '-' }"
    )
    print(f"  cost                           : {ledger}")
    return {
        "papers": len(records),
        "usable": len(usable),
        "rows": rows,
        "rows_with_dose": with_dose,
        "cost_usd": round(ledger.cost, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=int, default=100)
    parser.add_argument("--budget", type=float, default=20.0, help="USD ceiling")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--out", default="corpus.json")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    ledger = Ledger(args.budget)
    records = []
    for index, (pmcid, licence) in enumerate(search(args.papers, args.query), 1):
        if ledger.exhausted():
            print(f"  budget reached after {index - 1} papers", flush=True)
            break
        records.append(process(pmcid, licence, api_key, ledger))
        if index % 10 == 0:
            print(f"  ...{index} papers, {ledger}", flush=True)

    summary = report(records, ledger)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "papers": records}, handle, indent=2)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
