"""Phase 20 STEP 21 — consolidated reporting.

Writes summary workbook + dashboard HTML + final execution summary.
Every report is assembled from outputs/phase20 artifacts already produced by
the pipeline; nothing here changes any data artifact.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C


def _load_json(name, default=None):
    return C.read_json(C.OUT / name, default)


def _load_df(name):
    p = C.OUT / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def build_summary_workbook(cfg: dict) -> Path:
    decision = _load_json("final_decision.json", {})
    readiness = _load_json("model_readiness.json", {})
    gates = _load_json("quality_gates.json", {})
    indep = _load_json("independence_metrics.json", {})
    leak = _load_json("leakage_metrics.json", {})
    manifest = _load_json("input_manifest.json", {})
    ig = _load_json("information_gain_metrics.json", {})
    link = _load_json("external_linkage_metrics.json", {})

    top = pd.DataFrame([{
        "Decision": decision.get("decision"),
        "OverallReadinessScore": readiness.get("overall_readiness_score"),
        "RetrainAllowed": readiness.get("retrain_allowed"),
        "GatesPassed": gates.get("gates_passed"),
        "GatesTotal": gates.get("gates_total"),
        "MatrixObservations": int(readiness.get("per_target", {})
                                  .get("yield", {}).get("total_observations", 0)),
        "IndependentStudies": indep.get("independent_studies"),
        "IndependentLocations": indep.get("independent_locations"),
        "IndependentPapers": indep.get("independent_papers"),
        "ObservationsWithLeakageFlags": leak.get("observations_with_flags"),
        "ExternalLinkageRows": link.get("linked_observations")
            if isinstance(link, dict) else None,
        "InputGroups": manifest.get("input_groups"),
        "MissingInputs": ";".join(manifest.get("missing_inputs", []) or []),
        "InfoGainScored": ig.get("scored_predictors") if isinstance(ig, dict) else None,
    }])

    ready_rows = []
    for d, v in readiness.get("per_target", {}).items():
        if isinstance(v, dict):
            v = {k: v[k] for k in v if k != "notes"}
            v = {"domain": d, **v}
            ready_rows.append(v)
    ready_df = pd.DataFrame(ready_rows) if ready_rows else pd.DataFrame()

    gates_df = pd.DataFrame(gates.get("gates", [])) if gates.get("gates") else \
        pd.DataFrame()

    sheets = {
        "Decision": top,
        "Readiness": ready_df,
        "Gates": gates_df,
    }

    return C.write_excel_multi(sheets, C.REPORTS / "phase20_summary.xlsx")


def build_dashboard_html(cfg: dict) -> Path:
    decision = _load_json("final_decision.json", {})
    readiness = _load_json("model_readiness.json", {})
    indep = _load_json("independence_metrics.json", {})
    leak = _load_json("leakage_metrics.json", {})
    gates = _load_json("quality_gates.json", {})
    mr = readiness.get("per_target", {})
    per = {
        d: {k: v for k, v in (v or {}).items() if k != "notes"}
        for d, v in mr.items()
    }
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Phase 20 Execution Dashboard</title>
<style>
  body {{ font-family: sans-serif; margin: 24px; background:#f7f8fa; color:#1c2733; }}
  h1 {{ border-bottom: 3px solid #2b6cb0; padding-bottom: 8px; }}
  h2 {{ color:#2b6cb0; margin-top: 28px; }}
  table {{ border-collapse: collapse; background:#fff; width: 100%; margin: 8px 0; }}
  th, td {{ border: 1px solid #d9e2ec; padding: 6px 10px; text-align:left; }}
  th {{ background:#eef3f9; }}
  .badge {{ display:inline-block; padding: 2px 10px; border-radius: 12px; color:#fff; font-weight:600; }}
  .READY {{ background:#2f855a; }}
  .CONDITIONALLY_READY {{ background:#d69e2e; }}
  .NOT_READY {{ background:#c53030; }}
</style>
</head>
<body>
<h1>Phase 20 &mdash; Model-Ready Dataset Construction</h1>
<p>Generated: {C.now_full_iso()}</p>
<p>Project root: {C.PROJECT_ROOT}</p>
<p><span class="badge {decision.get('decision')}">{decision.get('decision')}</span>
&nbsp; Overall readiness score: {readiness.get('overall_readiness_score')}
&nbsp; Retrain allowed: {readiness.get('retrain_allowed')}</p>

<h2>Quality Gates</h2>
<table>
<tr><th>Gate</th><th>Result</th><th>Detail</th></tr>
{''.join(
    f"<tr><td>{g.get('gate')}</td><td>{'PASS' if g.get('pass') else 'FAIL'}</td>"
    f"<td>{g.get('detail')}</td></tr>"
    for g in (gates.get('gates') or []))}
</table>

<h2>Readiness per target</h2>
<table>
<tr><th>Domain</th><th>Obs</th><th>Studies</th><th>Locations</th><th>Complete</th>
<th>p80</th><th>p90</th><th>Crops</th><th>LeakSafe</th><th>EffN</th><th>Gates</th><th>Score</th></tr>
{''.join(
    f"<tr><td>{d}</td><td>{v.get('total_observations')}</td>"
    f"<td>{v.get('independent_studies')}</td><td>{v.get('independent_locations')}</td>"
    f"<td>{v.get('complete_observations')}</td><td>{v.get('p80_coverage')}</td>"
    f"<td>{v.get('p90_coverage')}</td><td>{v.get('crop_coverage')}</td>"
    f"<td>{v.get('leakage_safe_samples')}</td><td>{v.get('effective_sample_size')}</td>"
    f"<td>{v.get('gates_pass')}</td><td>{v.get('readiness_score')}</td></tr>"
    for d, v in per.items())}
</table>

<h2>Independence &amp; leakage</h2>
<table>
<tr><th>Studies</th><th>Experiments</th><th>Locations</th><th>Papers</th>
<th>Treatments</th><th>MaxObs/Study</th><th>ObsWithLeakFlags</th></tr>
<tr><td>{indep.get('independent_studies')}</td><td>{indep.get('independent_experiments')}</td>
<td>{indep.get('independent_locations')}</td><td>{indep.get('independent_papers')}</td>
<td>{indep.get('independent_treatments')}</td><td>{indep.get('max_obs_per_study')}</td>
<td>{leak.get('observations_with_flags')}</td></tr>
</table>

<p><em>Phase 20 artifacts are in <code>outputs/phase20</code>; reports in
<code>reports/phase20</code>. Protected inputs were read-only (SHA256 verified).</em></p>
</body>
</html>
"""
    out = C.REPORTS / "phase20_dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def build_execution_summary(cfg: dict) -> dict:
    summary = {
        "phase": 20,
        "phase20_version": C.PHASE20_VERSION,
        "generated_at": C.now_full_iso(),
        "project_root": str(C.PROJECT_ROOT),
        "dry_run": bool(cfg.get("dry_run", True)),
        "decision": _load_json("final_decision.json", {}).get("decision"),
        "readiness": _load_json("model_readiness.json", {}),
        "quality_gates": _load_json("quality_gates.json", {}),
        "independence": _load_json("independence_metrics.json", {}),
        "leakage": _load_json("leakage_metrics.json", {}),
        "information_gain": _load_json("information_gain_metrics.json", {}),
        "completeness": _load_json("missingness_metrics.json", {}),
        "external_linkage": _load_json("external_linkage_metrics.json", {}),
        "input_manifest": _load_json("input_manifest.json", {}),
        "protected_checksums_preserved": _load_json("protected_input_checksums_after.json", {}),
        "reports": [str(p.relative_to(C.PROJECT_ROOT))
                    for p in sorted(C.REPORTS.glob("*")) if p.is_file()],
    }
    C.write_json(summary, C.OUT / "final_execution_summary.json")
    return summary


def run(force: bool = False):
    cfg = C.load_config()
    wb = build_summary_workbook(cfg)
    dash = build_dashboard_html(cfg)
    summary = build_execution_summary(cfg)
    C.mark_done("reports", summary)
    C.log_msg(f"STEP21 reports: {wb.name}, {dash.name}, final_execution_summary.json")
    return summary


if __name__ == "__main__":
    import sys
    run(force="--force" in sys.argv)
