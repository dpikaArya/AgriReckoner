"""Re-derive ontology mappings by searching each authority for the column's LABEL.

The registry's identifiers were generated rather than resolved, so most pointed at an
unrelated concept that merely sat nearby in the identifier space (Yield_per_Hectare ->
Yunnan, Ash -> donkeys). Guessing replacement identifiers would repeat the mistake, so
every candidate here comes back from a label search and is kept only when the authority's
own prefLabel matches the column. Anything that does not match is DROPPED: for a
``skos:exactMatch``, asserting nothing is strictly better than asserting something false.

Writes the repaired registry to --out (default: in place with --write).
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

REGISTRY = Path(__file__).parent.parent / "spec" / "uams_ontology.yaml"
AGROVOC_SEARCH = "https://agrovoc.fao.org/browse/rest/v1/agrovoc/search"
OLS_SEARCH = "https://www.ebi.ac.uk/ols4/api/search"
TIMEOUT = 45
ACCEPT = 0.90  # label agreement required before an exactMatch is asserted

# Column names that are project shorthand rather than the term an authority would use.
QUERY_ALIASES = {
    "Yield_per_Hectare": "crop yield",
    "Yield_per_Acre": "crop yield",
    "Yield_per_Plot": "crop yield",
    "Plant_Height_cm": "plant height",
    "Leaf_Area_cm2": "leaf area",
    "Stem_Diameter_mm": "stem diameter",
    "Fruit_Diameter_mm": "fruit diameter",
    "100_Seed_Weight": "seed weight",
    "Soil_pH": "soil pH",
    "Organic_Carbon": "soil organic carbon",
    "Growing_Degree_Days": "degree days",
    "Growth_Duration_Days": "growing period",
    "Spacing_Plant": "plant spacing",
    "Fertilizer_Name": "fertilizers",
    "Scientific_Name": "taxonomic names",
    "Nitrogen_Use_Efficiency": "nutrient use efficiency",
    "Water_Use_Efficiency": "water use efficiency",
    "Phosphorus_Content": "phosphorus",
    "Iron_Content": "iron",
    "Sulphur_Content": "sulfur",
    "Protein": "protein content",
    "Fiber": "fibre content",
    "Ash": "ash content",
}


def _norm(text):
    return re.sub(r"[^a-z0-9 ]", " ", str(text).replace("_", " ").lower()).split()


def agreement(column, label):
    """1.0 when the authority's label says the same thing as the column name.

    Containment alone is not agreement: "nodes" appears inside "SE.02 two nodes or
    internodes visible stage", which is a growth stage rather than a node count. A subset
    is therefore only accepted when the label adds at most one word, and never when it
    carries stage or process wording.
    """
    left, right = set(_norm(QUERY_ALIASES.get(column, column))), set(_norm(label))
    if not left or not right:
        return 0.0
    # Nothing short of the same words is an exactMatch. One extra qualifier changes the
    # concept — "acid rainfall" is not rainfall, and "two nodes visible stage" is not a
    # node count — and it was exactly this latitude that produced the original bad map.
    return 1.0 if left == right else 0.0


def search_agrovoc(query):
    url = f"{AGROVOC_SEARCH}?query={urllib.parse.quote(query)}&lang=en&maxhits=8"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except Exception:
        return []
    return [
        (f"AGROVOC:{r['uri'].rsplit('/', 1)[-1]}", r.get("prefLabel", ""))
        for r in payload.get("results", [])
    ]


def search_ols(query, ontology):
    url = f"{OLS_SEARCH}?q={urllib.parse.quote(query)}&ontology={ontology.lower()}&rows=8"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            payload = json.load(response)
    except Exception:
        return []
    out = []
    for doc in payload.get("response", {}).get("docs", []):
        short = doc.get("obo_id") or doc.get("short_form", "")
        if short:
            out.append((short.replace("_", ":", 1), doc.get("label", "")))
    return out


def best_match(column, authority):
    query = QUERY_ALIASES.get(column, column.replace("_", " "))
    candidates = search_agrovoc(query) if authority == "AGROVOC" else search_ols(query, authority)
    scored = [(agreement(column, label), curie, label) for curie, label in candidates]
    scored.sort(reverse=True)
    return scored[0] if scored and scored[0][0] >= ACCEPT else None


def repair(registry_path, out_path, write):
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    columns = data.get("columns", data)
    repaired = dropped = kept = 0

    for column, spec in columns.items():
        if not isinstance(spec, dict) or not spec.get("terms"):
            continue
        new_terms = []
        for term in spec["terms"]:
            curie = term.get("curie", "")
            authority = curie.split(":", 1)[0] if ":" in curie else ""
            if authority not in {"AGROVOC", "PO", "ENVO", "FOODON"}:
                new_terms.append(term)
                kept += 1
                continue
            match = best_match(column, authority)
            if match:
                score, new_curie, label = match
                term = dict(term, curie=new_curie, label=label)
                new_terms.append(term)
                repaired += 1
                print(
                    f"  repaired {column:26s} {curie:18s} -> {new_curie:16s} {label} ({score:.2f})"
                )
            else:
                dropped += 1
                print(f"  DROPPED  {column:26s} {curie:18s} (no authority term matches the column)")
        spec["terms"] = new_terms

    print(f"\n{repaired} repaired, {dropped} dropped, {kept} left untouched")
    if write:
        out_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        print(f"wrote {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the registry in place")
    parser.add_argument("--out", type=Path, default=REGISTRY)
    args = parser.parse_args()
    return repair(REGISTRY, args.out, args.write)


if __name__ == "__main__":
    sys.exit(main())
