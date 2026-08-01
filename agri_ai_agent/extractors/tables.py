"""Reading experimental results out of a paper's tables.

Every rule here was established by auditing extraction against the source papers, and each
one corrects a failure that produced confident, wrong numbers rather than missing ones:

* Spans must be expanded, or a value lands in the wrong column.
* The treatment column must be found, not assumed: tables are commonly keyed by year or
  site first, so column 0 is often the year.
* A column is only a yield if its own header says so. "grain" and "harvest" also match
  "1000-grain weight" and "harvest index"; reading those as yield was the single largest
  error class measured.
* A row label must be a treatment, not a statistic. Agronomy tables append an analysis-of-
  variance block whose rows (M, N, M x N, CD, CV) carry F values shaped exactly like yields.
* Thousands separators are common above 1000 kg/ha. Mis-parsing them removes or corrupts
  precisely the highest-yielding rows, which biases a corpus rather than just shrinking it.

The functions are deliberately free of any model call: something else decides *which* table
and column to read, and this module decides whether that choice is credible and then reads
the numbers.
"""

import re

NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")

# A yield column names a yield. "grain" and "harvest" alone are too loose — they also match
# grain weight and harvest index, which are different quantities on a different scale.
YIELD_WORD = re.compile(r"\b(yield|gy|produc(?:tion|tivity)|output)\b", re.I)
AREA_UNIT = re.compile(r"\b(kg|t|q|mg|g)\s*[./·]?\s*(ha|hm|m\s*[-−]?\s*2|plot|plant|pot)\b", re.I)
DOSE_HEADER = re.compile(r"\b(rate|applied|application|dose|dosage|level|added|amount)\b", re.I)

# Captions whose subject is an analysis rather than an experiment. Split in two because a
# real results table often *also* reports a fitted curve or a calibration alongside its
# treatments: "Effect of irrigation on maize grain yield and fitting curve" is an experiment.
NEVER_AN_EXPERIMENT = re.compile(
    r"\b(correlation|regression|response surface|rmse|predicted vs|membership|entropy"
    r"|weight coefficient|analysis of variance|anova|sums? of squares"
    r"|model\s+(?:evaluation|performance|parameter)|simulat)\w*",
    re.I,
)
# These only disqualify a caption that does not otherwise announce a yield.
ANALYSIS_UNLESS_YIELD = re.compile(r"\b(fitted|fitting|calibrat|sensitivity|r²)\w*", re.I)


def caption_is_an_analysis(caption):
    """True when the caption's subject is an analysis rather than the trial's results."""
    if NEVER_AN_EXPERIMENT.search(caption):
        return True
    return bool(ANALYSIS_UNLESS_YIELD.search(caption)) and not YIELD_WORD.search(caption)


SUMMARY_LABEL = re.compile(
    r"^\s*(?:"
    r"(?:mean|range|cv|c\.?d\.?|lsd|total|average|treatments?|source|significance"
    r"|anova|ns|contrast|interaction|rmse|df|error|residual|block|replication)\b"
    r"|(?:main|simple)\s+effects?"
    r"|[fp]\s*[-–]\s*\w"
    r"|[fp]\s*[-–]?\s*value"
    r"|r\s*[²2]\b|η|χ|σ"
    r")",
    re.I,
)
# Two-letter codes that are a statistic at the foot of a table and a treatment elsewhere:
# "SD" meant straw deep incorporation in one trial, and was its highest-yielding treatment.
AMBIGUOUS_STAT = re.compile(r"^\s*(sd|se|cv|cd|ns|lsd|sem)\s*$", re.I)
FACTOR_TERM = re.compile(r"^\s*[A-Za-z]{1,3}\s*(?:[x×*]\s*[A-Za-z]{1,3}\s*)+$")
BARE_FACTOR = re.compile(r"^\s*[A-Za-z]\s*$")
BARE_YEAR = re.compile(r"^\s*(19|20)\d\d\s*$")
MEASURED_VARIABLE = re.compile(
    r"\((?:cm|mm|m|g|kg|t|q|%|n\.?\s*m|kg\s*h[lL]|°c|days?|no\.?)\b[^)]*\)?", re.I
)

FOOT_ROWS = 3  # rows at the end of a table where an ANOVA block is expected
MIN_TREATMENT_SHARE = 0.6  # below this the column is not a treatment column


def cell_text(node):
    """Flatten an XML cell to text, normalising the unicode minus and thin space."""
    text = "".join(node.itertext()).replace("−", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_grid(table_el):
    """Return (caption, rows) for a ``table-wrap`` element, with spans expanded.

    Spans are near-universal in these tables. Leaving them unexpanded shifts every
    subsequent cell one column left, so the numbers read are real numbers from the wrong
    variable — the hardest kind of error to notice downstream.
    """
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


def parse_measurement(text):
    """Read the measured value from a cell, dropping dispersion and significance letters.

    Handles "1 446" and "1,446": above a tonne per hectare a separator is normal, and
    mishandling it corrupts or deletes the highest-yielding rows specifically.
    """
    head = re.split(r"[±]", str(text))[0].strip()
    head = re.sub(r"[a-zA-Z\s*†‡]+$", "", head).strip()
    head = re.sub(r"(?<=\d)[\s,](?=\d{3}\b)", "", head)
    return float(head) if NUMERIC.match(head) else None


def is_treatment_label(label, near_foot=False):
    """True when a row label names an experimental treatment rather than a statistic.

    A dose written into the label ("N2 (300 kg/ha)") is a treatment; a measured variable
    ("Root length (cm)") is not. They differ by whether a number precedes the unit.
    """
    if AMBIGUOUS_STAT.match(label):
        return not near_foot
    if SUMMARY_LABEL.match(label):
        return False
    if FACTOR_TERM.match(label) or BARE_FACTOR.match(label) or BARE_YEAR.match(label):
        return False
    match = MEASURED_VARIABLE.search(label)
    return not (match and not re.search(r"\d\s*[a-zA-Z%]", match.group(0)))


def find_treatment_column(header, body):
    """Index of the column holding treatment labels.

    Never assume column 0: keying on it reads the year in every table that reports several
    seasons, which silently replaces the experimental variable with a date.
    """
    head = header[0] if header else []
    for index, name in enumerate(head):
        if re.search(r"\btreatments?\b|\bfertili[sz]er\b|\btreat\b", name, re.I):
            return index
    best, best_hits = 0, 0
    for index in range(max((len(r) for r in body), default=1)):
        hits = sum(
            1
            for row in body
            if len(row) > index
            and is_treatment_label(row[index])
            and not BARE_YEAR.match(row[index])
        )
        if hits > best_hits:
            best, best_hits = index, hits
    return best


def find_yield_column(header, caption=""):
    """Index of the column that reports a yield, or None when no column does."""
    head = header[0] if header else []
    for index, name in enumerate(head):
        if YIELD_WORD.search(name) and (AREA_UNIT.search(name) or AREA_UNIT.search(caption)):
            return index
    for index, name in enumerate(head):
        if YIELD_WORD.search(name):
            return index
    return None


def selection_problem(treatment_column, yield_column, header, caption, unit=""):
    """Return why a chosen table/column pair is not credible, or None when it is.

    The choice of table and column is checked against the table's own header rather than
    trusted, because reading a plausible number from the wrong column is indistinguishable
    from success without this check.
    """
    if treatment_column is None or yield_column is None:
        return "no usable column"
    if treatment_column < 0 or yield_column < 0:
        return "no usable column"
    if caption_is_an_analysis(caption):
        return f"caption describes an analysis, not an experiment: {caption[:60]}"
    head = header[0] if header else []
    if yield_column >= len(head):
        return "yield column is outside the header"
    cell = head[yield_column]
    names_yield = YIELD_WORD.search(cell)
    # A caption mentioning yield is not licence to read any column with an area unit: the
    # applied-rate column carries kg/ha too, and reading it returns a dose as a yield.
    inferred = (
        YIELD_WORD.search(caption) and AREA_UNIT.search(cell) and not DOSE_HEADER.search(cell)
    )
    if not names_yield and not inferred:
        return f"chosen column does not name a yield: {cell[:50]!r}"
    if not AREA_UNIT.search(f"{cell} {caption}") and not AREA_UNIT.search(unit or ""):
        return f"no area unit for the chosen column: {cell[:50]!r}"
    return None


def valid_dose_columns(dose_columns, header, yield_column):
    """Keep only dose columns whose header names an applied rate per unit area.

    Measured against real papers, most proposed dose columns were costs, grain weights, plot
    counts or the yield column itself. A wrong dose is worse than a missing one, because it
    is the variable a fertiliser recommendation would be built on.
    """
    head = header[0] if header else []
    keep = []
    for column in dose_columns or []:
        if not 0 <= column < len(head) or column == yield_column:
            continue
        cell = head[column]
        if DOSE_HEADER.search(cell) and AREA_UNIT.search(cell) and not YIELD_WORD.search(cell):
            keep.append(column)
    return keep


def _foot_block_start(body, treatment_column):
    """Index where the trailing statistics block begins, or len(body) if there is none.

    An ANOVA block runs to the end of the table once it starts, so the foot is the longest
    run of unambiguous statistic rows at the bottom. Taking a fixed number of rows instead
    would classify most of a short table as its own footer.
    """
    run_start, saw_unambiguous, run_length = len(body), False, 0
    for index in range(len(body) - 1, -1, -1):
        row = body[index]
        if treatment_column >= len(row) or not row[treatment_column].strip():
            continue
        label = row[treatment_column].strip()
        unambiguous = bool(
            SUMMARY_LABEL.match(label) or FACTOR_TERM.match(label) or BARE_FACTOR.match(label)
        )
        if unambiguous or AMBIGUOUS_STAT.match(label):
            run_start, run_length = index, run_length + 1
            saw_unambiguous = saw_unambiguous or unambiguous
            continue
        break
    # A lone trailing "SD" beside real treatments is undecidable; a run of statistics, or
    # any unambiguous one, marks where the analysis block starts.
    return run_start if (saw_unambiguous or run_length >= 2) else len(body)


def read_rows(body, treatment_column, yield_column, dose_columns=(), unit=""):
    """Read treatment rows from a table body, keeping the source cell for each value."""
    foot_starts = _foot_block_start(body, treatment_column)
    rows, candidates = [], 0
    for row_index, row in enumerate(body):
        if treatment_column >= len(row) or yield_column >= len(row):
            continue
        label = row[treatment_column].strip()
        value = parse_measurement(row[yield_column])
        if not label or value is None:
            continue
        candidates += 1
        if not is_treatment_label(label, near_foot=row_index >= foot_starts):
            continue
        rows.append(
            {
                "treatment": label,
                "yield_value": value,
                "yield_unit": unit,
                "doses": [parse_measurement(row[c]) for c in dose_columns if c < len(row)],
                "row_index": row_index,
                "source_cell": row[yield_column],
            }
        )
    # Mostly non-treatment labels means the column is not a treatment column, usually
    # because the table is transposed with measured variables down the side.
    if candidates and len(rows) / candidates < MIN_TREATMENT_SHARE:
        return []
    return rows


if __name__ == "__main__":
    import xml.etree.ElementTree as ET

    xml = """<table-wrap><label>Table 6</label><caption><p>Grain yield by treatment</p></caption>
    <table>
    <tr><th>Year</th><th>Treatment</th><th>N applied (kg/ha)</th><th>Grain yield (kg ha-1)</th></tr>
    <tr><td rowspan="2">2021</td><td>T1</td><td>0</td><td>1 446ab</td></tr>
    <tr><td>T2</td><td>120</td><td>2,884a</td></tr>
    <tr><td>2022</td><td>SD</td><td>60</td><td>6173.40 b</td></tr>
    <tr><td>Mean</td><td></td><td></td><td>3501.13</td></tr>
    </table></table-wrap>"""
    caption, grid = parse_grid(ET.fromstring(xml))
    header, body = grid[:1], grid[1:]
    treat = find_treatment_column(header, body)
    yields = find_yield_column(header, caption)
    assert treat == 1, treat  # column 0 is the year
    assert yields == 3, yields
    assert selection_problem(treat, yields, header, caption) is None
    doses = valid_dose_columns([2], header, yields)
    rows = read_rows(body, treat, yields, doses, "kg ha-1")
    assert [r["treatment"] for r in rows] == ["T1", "T2", "SD"], rows
    assert [r["yield_value"] for r in rows] == [1446.0, 2884.0, 6173.40], rows
    assert rows[0]["doses"] == [0.0]
    assert selection_problem(1, 2, header, caption) is not None  # dose column is not a yield
    print("tables smoke OK ->", [(r["treatment"], r["yield_value"]) for r in rows])
