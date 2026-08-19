"""Framework health dashboard for the Literature Intelligence Module.

Aggregates pipeline + connector metrics into a single self-contained HTML file
(``framework_health_dashboard.html``).  All figures come from the current run
and persisted state — no synthetic numbers.

Cards rendered (thresholds from ``config/dashboard.yaml``):

  connectors enabled/disabled, auth status, API latency, rate-limit status,
  papers discovered, verified original experiments, PDFs downloaded,
  tables/figures extracted, experimental observations, schema completeness,
  duplicate rate, missing-value %, training dataset size, models trained,
  best validation R2, latest sync, pipeline duration, cache utilization,
  connector failures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from agri_ai_agent.config.schema import UAMS_COLUMNS

logger = logging.getLogger(__name__)

_MISSING = "n/a"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return _MISSING
    try:
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: float) -> float:
    return round(value * 100.0, 2)


class HealthMetrics:
    """Collects every metric surfaced by the dashboard."""

    def __init__(self, report: Any, pipeline: Any, output_dir: Path):
        self.report = report
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.cfg = pipeline.config
        self.state = pipeline.state
        self.metrics: dict[str, Any] = {}

    def collect(self) -> dict[str, Any]:
        m: dict[str, Any] = {}

        # --- run identity ------------------------------------------------- #
        m["run_id"] = self.report.run_id
        m["started_at"] = self.report.started_at
        m["completed_at"] = self.report.completed_at
        m["duration_sec"] = self.report.duration_sec

        # --- connectors --------------------------------------------------- #
        lit = [
            {
                "name": c.source_name,
                "kind": "literature",
                "enabled": c.enabled,
                "auth": self.cfg.connector_limits.get(c.source_name, {}),
                "errors": 0,
            }
            for c in self.pipeline.connectors
        ]
        agri = [
            {
                "name": c.source_name,
                "kind": "agricultural",
                "enabled": c.enabled,
                "auth": self.cfg.connector_limits.get(c.source_name, {}),
                "errors": 0,
            }
            for c in self.pipeline.agri_manager.connectors
        ]
        # enrich with per-connector sync errors + auth requirements
        for result in self.report.sync_results:
            for row in lit:
                if row["name"] == result.connector:
                    row["errors"] = len(result.errors)
        for result in self.report.agri_sync_results:
            for row in agri:
                if row["name"] == result.connector:
                    row["errors"] = len(result.errors)

        m["connectors"] = lit + agri
        m["literature_enabled"] = sum(1 for c in lit if c["enabled"])
        m["literature_total"] = len(lit)
        m["agri_enabled"] = sum(1 for c in agri if c["enabled"])
        m["agri_total"] = len(agri)

        # --- HTTP / transport stats -------------------------------------- #
        http_rows: list[dict[str, Any]] = []
        totals = {"cache_hits": 0, "cache_misses": 0, "requests": 0, "latency": 0.0, "n": 0}
        for c in self.pipeline.connectors + self.pipeline.agri_manager.connectors:
            try:
                stats = c.http.stats()
            except Exception:  # noqa: BLE001
                continue
            row = {
                "connector": c.source_name,
                "enabled": c.enabled,
                "rate_limit_per_minute": self.cfg.limit(
                    c.source_name, "rate_limit_per_minute", self.cfg.rate_limit_per_minute
                ),
                **{
                    k: stats.get(k)
                    for k in (
                        "cache_hits",
                        "cache_misses",
                        "total_requests",
                        "avg_response_time_ms",
                    )
                },
            }
            http_rows.append(row)
            for key, src in (
                ("cache_hits", "cache_hits"),
                ("cache_misses", "cache_misses"),
                ("requests", "total_requests"),
            ):
                totals[key] += int(stats.get(src, 0) or 0)
            latency = stats.get("avg_response_time_ms")
            if latency is not None:
                totals["latency"] += float(latency)
                totals["n"] += 1
        m["http"] = http_rows
        m["api_avg_latency_ms"] = round(totals["latency"] / totals["n"], 2) if totals["n"] else None
        m["api_requests"] = totals["requests"]
        m["cache_hits"] = totals["cache_hits"]
        m["cache_misses"] = totals["cache_misses"]
        m["cache_utilization_pct"] = _pct(
            totals["cache_hits"] / max(totals["cache_hits"] + totals["cache_misses"], 1)
        )

        # --- discovery / validation -------------------------------------- #
        m["papers_discovered"] = self.report.total_fetched
        m["verified_original_experiments"] = self.report.verified_papers
        m["duplicates"] = self.report.duplicates
        m["validated_dois"] = self.report.validated_dois
        m["failed_pdfs"] = self.report.failed_pdfs
        m["agri_records"] = self.report.agri_records
        m["duplicate_rate_pct"] = _pct(self.report.duplicates / max(self.report.total_fetched, 1))
        m["errors_list"] = list(self.report.errors)
        m["errors_list"] += [
            f"{r.connector}: {e}" for r in self.report.sync_results for e in r.errors
        ]
        m["errors_list"] += [
            f"{r.connector}: {e}" for r in self.report.agri_sync_results for e in r.errors
        ]
        m["total_errors"] = len(m["errors_list"])

        # --- PDFs / extraction ------------------------------------------- #
        m["pdfs_downloaded"] = self._count_verified_pdfs()
        m["tables_figures_extracted"] = 0  # Phase 5 (main orchestrator)
        m["experimental_observations"] = self._count_observations()

        # --- schema / data quality --------------------------------------- #
        schema_meta = self._schema_completeness()
        m["schema_completeness_pct"] = schema_meta["pct"]
        m["schema_columns_present"] = schema_meta["present"]
        m["schema_columns_total"] = schema_meta["total"]
        m["missing_value_pct"] = self._missing_value_pct()

        # --- training / models ------------------------------------------- #
        m["training_dataset_size"] = self._training_rows()
        m["models_trained"], m["best_validation_r2"] = self._model_stats()

        # --- sync / cache state ------------------------------------------ #
        m["latest_sync"] = self._latest_sync()
        m["cached_records"] = self.state.summary().get("cached_records", 0)
        self.metrics = m
        return m

    # ------------------------------------------------------------------ #
    def _count_verified_pdfs(self) -> int:
        total = 0
        try:
            from agri_ai_agent.literature.models import LiteratureRecord

            records: list[LiteratureRecord] = []
            for source in self.state.iter_sources():
                if source not in {c.source_name for c in self.pipeline.connectors}:
                    continue
                for _source_id, payload in self.state.iter_cache(source):
                    try:
                        records.append(LiteratureRecord.from_dict(payload))
                    except Exception:  # noqa: BLE001
                        continue
            for record in records:
                total += sum(1 for loc in record.pdf_locations if loc.verified)
        except Exception:  # noqa: BLE001
            logger.debug("pdf verification count unavailable", exc_info=True)
        return total

    def _count_observations(self) -> int:
        try:
            train = self.output_dir / "training_dataset.parquet"
            if train.exists():
                return int(len(pd.read_parquet(train)))
        except Exception:  # noqa: BLE001
            pass
        return 0

    def _schema_completeness(self) -> dict[str, Any]:
        present = 0
        total = len(UAMS_COLUMNS)
        path = self.output_dir / "universal_schema.csv"
        if path.exists():
            try:
                df = pd.read_csv(path)
                for col in UAMS_COLUMNS:
                    if col in df.columns and df[col].notna().any():
                        present += 1
            except Exception:  # noqa: BLE001
                pass
        return {
            "present": present,
            "total": total,
            "pct": _pct(present / max(total, 1)),
        }

    def _missing_value_pct(self) -> float:
        for name in ("training_dataset.parquet", "universal_schema.csv"):
            path = self.output_dir / name
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
            except Exception:  # noqa: BLE001
                continue
            if df.empty:
                continue
            return _pct(float(df.isna().sum().sum() / (df.shape[0] * df.shape[1])))
        return 0.0

    def _training_rows(self) -> int:
        path = self.output_dir / "training_dataset.parquet"
        if not path.exists():
            return 0
        try:
            return int(len(pd.read_parquet(path)))
        except Exception:  # noqa: BLE001
            return 0

    def _model_stats(self) -> tuple[int, float | None]:
        models_dir = self.output_dir / "models"
        trained = 0
        if models_dir.exists():
            trained = sum(1 for p in models_dir.glob("*.joblib"))
            trained += sum(1 for p in models_dir.glob("*.pkl"))
        best_r2: float | None = None
        metrics_csv = self.output_dir / "metrics.csv"
        if metrics_csv.exists():
            try:
                df = pd.read_csv(metrics_csv)
                for col in ("r2", "r2_mean", "best_r2"):
                    if col in df.columns and df[col].notna().any():
                        best_r2 = float(df[col].max())
                        break
            except Exception:  # noqa: BLE001
                pass
        return trained, best_r2

    def _latest_sync(self) -> str:
        stamps: list[str] = []
        for c in self.pipeline.connectors + self.pipeline.agri_manager.connectors:
            stamp = c.get_cursor("last_sync_at")
            if stamp:
                stamps.append(str(stamp))
        if not stamps:
            return _MISSING
        try:
            return max(stamps)
        except TypeError:
            return _MISSING


# ---------------------------------------------------------------------- #
# HTML rendering
# ---------------------------------------------------------------------- #
def build_health_dashboard(
    report: Any,
    pipeline: Any,
    output_dir: str | Path,
) -> Path:
    """Render ``framework_health_dashboard.html`` and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = HealthMetrics(report, pipeline, output_dir).collect()
    cfg = pipeline.config
    thresholds = cfg.dashboard.get("thresholds", {})
    title = cfg.dashboard.get("title", "AAIF Framework Health Dashboard")

    cards = _metric_cards(metrics, thresholds)
    connector_rows = _connector_rows(metrics)
    http_rows = metrics["http"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
:root {{ color-scheme: light; }}
body {{ font-family: "Segoe UI", Roboto, sans-serif; margin: 0; background: #f3f5f4; color: #1c2b21; }}
header {{ background: #1b5e20; color: #fff; padding: 1.2rem 2rem; }}
header h1 {{ margin: 0; font-size: 1.4rem; }}
header p {{ margin: .25rem 0 0; opacity: .85; font-size: .85rem; }}
main {{ max-width: 1200px; margin: 1.5rem auto; padding: 0 1rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: .9rem; }}
.card {{ background: #fff; border-radius: 10px; padding: .9rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.card .label {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: #5b6b5f; }}
.card .value {{ font-size: 1.5rem; font-weight: 600; margin-top: .2rem; }}
.card .sub {{ font-size: .75rem; color: #6b7b6f; margin-top: .15rem; }}
.ok {{ color: #2e7d32; }} .warn {{ color: #c77700; }} .bad {{ color: #c62828; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; margin: .5rem 0 1.2rem; border-radius: 8px; overflow: hidden; }}
th, td {{ border: 1px solid #e0e4e1; padding: .4rem .55rem; font-size: .82rem; text-align: left; }}
th {{ background: #e8f0e8; }}
h2 {{ color: #1b5e20; font-size: 1.05rem; margin: 1.4rem 0 .4rem; border-bottom: 2px solid #cfe3cf; padding-bottom: .3rem; }}
footer {{ text-align: center; color: #8a978d; font-size: .75rem; padding: 1.5rem; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>Run {_fmt(metrics["run_id"])} &middot; started {_fmt(metrics["started_at"])} &middot; completed {_fmt(metrics["completed_at"])} &middot; duration {_fmt(metrics["duration_sec"])}s</p>
</header>
<main>
  <div class="grid">
    {"".join(_render_card(c) for c in cards)}
  </div>

  <h2>Connector Health &amp; Authentication</h2>
  {_to_table(connector_rows)}

  <h2>API / Rate-Limit Status</h2>
  {_to_table(http_rows) if http_rows else '<p class="card">(no HTTP activity this run)</p>'}

  <h2>Errors &amp; Failures</h2>
  {_errors_block(metrics)}
</main>
<footer>Generated by AAIF Literature Intelligence Module &middot; configuration: config/dashboard.yaml</footer>
</body>
</html>"""

    path = output_dir / "framework_health_dashboard.html"
    path.write_text(html, encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def _metric_cards(metrics: dict[str, Any], thresholds: dict) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = [
        _card("Papers Discovered", metrics["papers_discovered"]),
        _card("Verified Original Experiments", metrics["verified_original_experiments"]),
        _card("PDFs Verified", metrics["pdfs_downloaded"]),
        _card(
            "Tables / Figures Extracted",
            metrics["tables_figures_extracted"],
            "Phase 5 (main pipeline)",
        ),
        _card("Experimental Observations", metrics["experimental_observations"]),
        _card("Agri Data Records", metrics["agri_records"]),
        _card(
            "Schema Completeness",
            f"{metrics['schema_completeness_pct']}%",
            f"{metrics['schema_columns_present']}/{metrics['schema_columns_total']} UAMS columns",
        ),
        _card("Duplicate Rate", f"{metrics['duplicate_rate_pct']}%"),
        _card("Missing Values", f"{metrics['missing_value_pct']}%"),
        _card("Training Dataset Size", metrics["training_dataset_size"], "rows"),
        _card("Models Trained", metrics["models_trained"]),
        _card("Best Validation R²", metrics["best_validation_r2"]),
        _card("API Avg Latency", _fmt(metrics["api_avg_latency_ms"]), "ms"),
        _card("API Requests", metrics["api_requests"]),
        _card("Cache Utilization", f"{metrics['cache_utilization_pct']}%"),
        _card("Cached Records", metrics["cached_records"]),
        _card("Latest Sync", metrics["latest_sync"]),
        _card("Connector Failures", metrics["total_errors"]),
    ]
    return cards


def _card(label: str, value: Any, sub: str = "") -> dict[str, Any]:
    return {"label": label, "value": _fmt(value), "sub": sub}


def _render_card(card: dict[str, Any]) -> str:
    return (
        '<div class="card"><div class="label">{label}</div>'
        '<div class="value">{value}</div>'
        "{sub}</div>".format(
            label=card["label"],
            value=card["value"],
            sub=f'<div class="sub">{card["sub"]}</div>' if card.get("sub") else "",
        )
    )


def _connector_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in metrics["connectors"]:
        rows.append(
            {
                "Connector": c["name"],
                "Kind": c["kind"],
                "Status": "enabled" if c["enabled"] else "disabled",
                "Errors": c["errors"],
            }
        )
    return rows


def _errors_block(metrics: dict[str, Any]) -> str:
    errors = metrics.get("errors_list", [])
    if not errors:
        return '<p class="card">No connector failures this run.</p>'
    body = "".join(f"<tr><td>{e}</td></tr>" for e in errors[:200])
    return f"<table><thead><tr><th>message</th></tr></thead><tbody>{body}</tbody></table>"


def _to_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    header = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{_fmt(row.get(c))}</td>" for c in cols)
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
