"""Check every ontology mapping against the authority that owns the term.

A ``skos:exactMatch`` is a strong, transitive claim: anyone consuming the registry inherits
it, and a wrong one propagates silently into their graph. This script resolves each CURIE
against its authority and compares the returned label to the column it is attached to, so a
mapping can never be published on the strength of a plausible-looking identifier alone.

Run offline-safe: with --offline it validates syntax and the local cache only.
Exit code 1 when any mapping resolves to a label unrelated to its column.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import yaml

REGISTRY = Path(__file__).parent.parent / "spec" / "uams_ontology.yaml"
CACHE = Path(__file__).parent.parent / "spec" / "ontology_labels.json"
AGROVOC_SPARQL = "https://agrovoc.fao.org/sparql"
OLS_BASE = "https://www.ebi.ac.uk/ols4/api/ontologies"
SIMILARITY_FLOOR = 0.55  # below this the label and the column name are unrelated
TIMEOUT = 45


def load_terms(registry_path=REGISTRY):
    """Return [(column, predicate, curie)] for every mapping in the registry."""
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    columns = data.get("columns", data)
    terms = []
    for column, spec in columns.items():
        if not isinstance(spec, dict):
            continue
        for term in spec.get("terms") or []:
            curie = term.get("curie", "")
            if ":" in curie:
                terms.append((column, term.get("predicate", ""), curie))
    return terms


def _sparql(codes):
    values = " ".join(f"<http://aims.fao.org/aos/agrovoc/{c}>" for c in codes)
    query = (
        f"SELECT ?s ?l WHERE {{ VALUES ?s {{ {values} }} "
        "?s <http://www.w3.org/2004/02/skos/core#prefLabel> ?l . FILTER(lang(?l)='en') }"
    )
    url = f"{AGROVOC_SPARQL}?query={urllib.parse.quote(query)}"
    request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        payload = json.load(response)
    return {
        binding["s"]["value"].rsplit("/", 1)[-1]: binding["l"]["value"]
        for binding in payload["results"]["bindings"]
    }


def resolve_agrovoc(codes, batch=40):
    labels = {}
    codes = list(codes)
    for start in range(0, len(codes), batch):
        try:
            labels.update(_sparql(codes[start : start + batch]))
        except Exception as exc:  # network/endpoint problems must not look like bad mappings
            print(f"  ! AGROVOC lookup failed for a batch: {exc}", file=sys.stderr)
    return labels


def resolve_ols(prefix, identifier):
    """Resolve an OBO-style term (PO, ENVO, FOODON) through OLS4."""
    iri = f"http://purl.obolibrary.org/obo/{prefix}_{identifier}"
    url = f"{OLS_BASE}/{prefix.lower()}/terms?iri={urllib.parse.quote(iri, safe='')}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            payload = json.load(response)
        return payload["_embedded"]["terms"][0]["label"]
    except Exception:
        return None


# Authorities use US spellings; the schema uses British ones. Same concept, different letters.
_SPELLING = (
    ("sulphur", "sulfur"),
    ("fibre", "fiber"),
    ("colour", "color"),
    ("aluminium", "aluminum"),
)


def _despell(text):
    for british, american in _SPELLING:
        text = text.replace(british, american)
    return text


def similarity(column, label):
    """How close a column name and an authority label are, ignoring separators."""
    left = _despell(column.replace("_", " ").lower())
    right = _despell((label or "").lower())
    if not right:
        return 0.0
    if left in right or right in left:
        return 1.0
    ratio = SequenceMatcher(None, left, right).ratio()
    shared = set(left.split()) & set(right.split())
    return max(ratio, 0.9 if shared else 0.0)


def verdicts(terms, labels):
    """Classify each mapping as ok, mismatch, or unresolved."""
    rows = []
    for column, predicate, curie in terms:
        label = labels.get(curie)
        if label is None:
            rows.append((column, predicate, curie, None, "unresolved", 0.0))
            continue
        score = similarity(column, label)
        status = "ok" if score >= SIMILARITY_FLOOR else "mismatch"
        rows.append((column, predicate, curie, label, status, score))
    return rows


def collect_labels(terms, offline=False):
    if offline:
        return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    agrovoc = [c.split(":", 1)[1] for _, _, c in terms if c.startswith("AGROVOC:")]
    labels = {f"AGROVOC:{k}": v for k, v in resolve_agrovoc(agrovoc).items()}
    for _, _, curie in terms:
        prefix, _, identifier = curie.partition(":")
        if prefix in {"PO", "ENVO", "FOODON"}:
            label = resolve_ols(prefix, identifier)
            if label:
                labels[curie] = label
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use the cached labels only")
    parser.add_argument("--write-cache", action="store_true", help="save resolved labels")
    args = parser.parse_args()

    terms = load_terms()
    labels = collect_labels(terms, offline=args.offline)
    if args.write_cache and labels:
        CACHE.write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")

    rows = verdicts(terms, labels)
    mismatched = [r for r in rows if r[4] == "mismatch"]
    unresolved = [r for r in rows if r[4] == "unresolved"]

    for column, predicate, curie, label, status, score in sorted(rows, key=lambda r: r[4]):
        if status == "ok":
            continue
        shown = label if label else "(unresolved)"
        print(f"  [{status:10s}] {column:28s} {curie:18s} -> {shown}  ({score:.2f}) {predicate}")

    print(
        f"\n{len(rows)} mappings: {len(rows) - len(mismatched) - len(unresolved)} ok, "
        f"{len(mismatched)} mismatched, {len(unresolved)} unresolved"
    )
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
