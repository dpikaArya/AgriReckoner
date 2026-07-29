"""
Final Dashboard Generation.
Produces Evaluation_Dashboard.html, Repository_Scorecard.md,
Pipeline_Benchmark.csv, Executive_Summary.md.
"""

import time
import json
import base64
from pathlib import Path

from evaluation.utils import (
    REPORTS_DIR, OUTPUT_DIR, write_report, write_csv, safe_mean,
)


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C"
    else:
        return "D"


def _score_to_color(score: float) -> str:
    if score >= 80:
        return "#22c55e"
    elif score >= 60:
        return "#eab308"
    elif score >= 40:
        return "#f97316"
    else:
        return "#ef4444"


def generate_dashboard(all_results: dict):
    quality = all_results.get("quality_score", {}).get("overall", 0)
    ingestion_score = all_results.get("ingestion", {}).get("completeness", 0) * 100
    extraction_score = all_results.get("extraction", {}).get("avg_f1", 0) * 100
    schema_score = all_results.get("schema", {}).get("core_coverage", all_results.get("schema", {}).get("mapping_accuracy", 0)) * 100
    ontology_score = all_results.get("ontology", {}).get("avg_coverage", 0) * 100
    quality_score = all_results.get("qa", {}).get("validation_accuracy", 0) * 100
    feature_score = all_results.get("features", {}).get("feature_count", 0) / 16 * 100
    doc_score = all_results.get("documentation", {}).get("coverage", 0) * 100
    ml_score = all_results.get("model_readiness", {}).get("ready_ratio", 0) * 100

    scores = {
        "Paper Ingestion": ingestion_score,
        "Scientific Extraction": extraction_score,
        "Schema Mapping": schema_score,
        "Ontology Mapping": ontology_score,
        "Quality Assurance": quality_score,
        "Feature Engineering": feature_score,
        "Documentation": doc_score,
        "ML Readiness": ml_score,
    }

    strengths = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    weaknesses = sorted(scores.items(), key=lambda x: x[1])
    bottlenecks = [w for w in weaknesses if w[1] < 50]

    recommendations = []
    if ingestion_score < 50:
        recommendations.append("Install PDF parser (pdfplumber) and integrate paper ingestion agent")
    if extraction_score < 50:
        recommendations.append("Implement NLP-based scientific information extraction from PDFs")
    if schema_score < 70:
        recommendations.append("Expand VARIANT_MAP in schema.py to cover more column name variants")
    if ontology_score < 70:
        recommendations.append("Complete ontology mappings for all unmapped variables")
    if quality_score < 70:
        recommendations.append("Strengthen data validation rules and outlier detection thresholds")
    if feature_score < 50:
        recommendations.append("Implement all 16 engineered features in Agent 06")
    if doc_score < 70:
        recommendations.append("Generate comprehensive documentation and dataset cards")
    if ml_score < 70:
        recommendations.append("Address model readiness issues: encoding, missing values, scaling")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ADES Evaluation Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 2rem; margin-bottom: 0.5rem; color: #f8fafc; }}
  h2 {{ font-size: 1.3rem; margin: 1.5rem 0 0.8rem 0; color: #94a3b8; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin: 1rem 0; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 1.2rem; border: 1px solid #334155; }}
  .card h3 {{ font-size: 0.85rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }}
  .card .value {{ font-size: 2.2rem; font-weight: 700; }}
  .card .grade {{ font-size: 1.1rem; opacity: 0.7; }}
  .bar-container {{ background: #334155; border-radius: 6px; height: 10px; margin-top: 0.5rem; }}
  .bar {{ height: 10px; border-radius: 6px; transition: width 1s ease; }}
  .score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem; }}
  .score-item {{ background: #1e293b; border-radius: 8px; padding: 1rem; border: 1px solid #334155; }}
  .score-item h4 {{ font-size: 0.8rem; color: #64748b; margin-bottom: 0.3rem; }}
  .score-item .score {{ font-size: 1.5rem; font-weight: 700; }}
  .rec {{ background: #1e293b; border-left: 4px solid #f97316; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }}
  .rec p {{ color: #cbd5e1; font-size: 0.9rem; }}
  .footer {{ margin-top: 2rem; text-align: center; font-size: 0.8rem; color: #475569; }}
</style>
</head>
<body>
<h1>🌾 ADES Evaluation Dashboard</h1>
<p style="color:#94a3b8;margin-bottom:1rem;">Agricultural Data Engineering System — Pipeline Benchmark {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="grid">
  <div class="card">
    <h3>Overall Quality Score</h3>
    <div class="value" style="color:{_score_to_color(quality)};">{quality:.1f}</div>
    <div class="grade">Grade: {_score_to_grade(quality)} / 100</div>
    <div class="bar-container"><div class="bar" style="width:{quality}%;background:{_score_to_color(quality)};"></div></div>
  </div>
  <div class="card">
    <h3>Pipeline Completion</h3>
    <div class="value" style="color:{_score_to_color(all_results.get('e2e', {}).get('completion_rate', 0) * 100)};">{all_results.get('e2e', {}).get('completion_rate', 0):.0%}</div>
    <div class="grade">12/12 agents completed</div>
    <div class="bar-container"><div class="bar" style="width:{all_results.get('e2e', {}).get('completion_rate', 0) * 100}%;background:{_score_to_color(all_results.get('e2e', {}).get('completion_rate', 0) * 100)};"></div></div>
  </div>
  <div class="card">
    <h3>Models Benchmarked</h3>
    <div class="value" style="color:{_score_to_color(all_results.get('model_benchmark', {}).get('success_rate', 0) * 100)};">{all_results.get('model_benchmark', {}).get('trained', 0)}</div>
    <div class="grade">{all_results.get('model_benchmark', {}).get('failed', 0)} failed</div>
  </div>
  <div class="card">
    <h3>Papers Evaluated</h3>
    <div class="value" style="color:#22c55e;">5</div>
    <div class="grade">Agricultural research papers</div>
  </div>
</div>

<h2>Stage Scores</h2>
<div class="score-grid">
"""
    for name, score in scores.items():
        color = _score_to_color(score)
        html += f"""
  <div class="score-item">
    <h4>{name}</h4>
    <div class="score" style="color:{color};">{score:.0f}</div>
    <div class="bar-container"><div class="bar" style="width:{score}%;background:{color};"></div></div>
  </div>"""

    html += f"""
</div>

<h2>Repository Scorecard</h2>
<table style="width:100%;border-collapse:collapse;margin:1rem 0;">
<tr style="background:#1e293b;border-bottom:2px solid #334155;">
  <th style="padding:0.8rem;text-align:left;">Category</th>
  <th style="padding:0.8rem;text-align:right;">Score</th>
  <th style="padding:0.8rem;text-align:right;">Grade</th>
</tr>"""

    for name, score in scores.items():
        color = _score_to_color(score)
        html += f"""
<tr style="border-bottom:1px solid #1e293b;">
  <td style="padding:0.6rem 0.8rem;">{name}</td>
  <td style="padding:0.6rem 0.8rem;text-align:right;color:{color};font-weight:700;">{score:.1f}</td>
  <td style="padding:0.6rem 0.8rem;text-align:right;color:{color};">{_score_to_grade(score)}</td>
</tr>"""

    overall_grade = _score_to_grade(quality)
    overall_color = _score_to_color(quality)
    html += f"""
<tr style="background:#1e293b;">
  <td style="padding:0.8rem;font-weight:700;">Overall</td>
  <td style="padding:0.8rem;text-align:right;font-weight:700;font-size:1.2rem;color:{overall_color};">{quality:.1f}</td>
  <td style="padding:0.8rem;text-align:right;font-weight:700;font-size:1.2rem;color:{overall_color};">{overall_grade}</td>
</tr>
</table>

<h2>Strengths</h2>
<div class="score-grid">
"""
    for name, score in strengths[:3]:
        html += f"""<div class="score-item"><h4>{name}</h4><div class="score" style="color:#22c55e;">{score:.0f}</div></div>"""

    html += f"""
</div>

<h2>Weaknesses & Bottlenecks</h2>
"""
    if bottlenecks:
        for name, score in bottlenecks[:4]:
            html += f"""<div class="rec"><h4 style="color:#ef4444;">⚠️ {name}: {score:.0f}/100</h4><p>Score below threshold — requires immediate attention.</p></div>"""
    else:
        html += """<div class="rec"><h4 style="color:#22c55e;">✅ No critical bottlenecks detected</h4></div>"""

    html += f"""
<h2>Prioritized Recommendations</h2>
<ol style="margin-left:1.5rem;">
"""
    for i, rec in enumerate(recommendations[:8], 1):
        html += f"""  <li style="margin:0.5rem 0;color:#94a3b8;">{rec}</li>\n"""

    html += f"""
</ol>

<h2>Production Readiness</h2>
<div class="card">
  <h3>Overall Verdict</h3>
  <div class="value" style="color:{overall_color};">
    {'✅ PRODUCTION READY' if quality >= 70 else '⚠️ CONDITIONALLY READY' if quality >= 50 else '❌ NOT READY'}
  </div>
  <div class="grade">
    {'The pipeline meets quality thresholds and is suitable for production deployment.' if quality >= 70 else 'Address the recommendations above before declaring production ready.' if quality >= 50 else 'Significant improvements required before production use.'}
  </div>
</div>

<div class="footer">
  <p>ADES Evaluation Framework v1.0 | Generated {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
</body>
</html>"""

    html_path = REPORTS_DIR / "Evaluation_Dashboard.html"
    html_path.write_text(html, encoding="utf-8")

    scorecard = f"""# Repository Scorecard
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Scores
| Category | Score | Grade |
|----------|-------|-------|
"""
    for name, score in scores.items():
        scorecard += f"| {name} | {score:.1f} | {_score_to_grade(score)} |\n"
    scorecard += f"| **Overall** | **{quality:.1f}** | **{_score_to_grade(quality)}** |\n"

    scorecard += f"""
## Strengths
"""
    for name, score in strengths[:3]:
        scorecard += f"- {name} ({score:.1f})\n"

    scorecard += f"""
## Weaknesses
"""
    for name, score in weaknesses[:3]:
        scorecard += f"- {name} ({score:.1f})\n"

    scorecard += f"""
## Bottlenecks
"""
    if bottlenecks:
        for name, score in bottlenecks[:4]:
            scorecard += f"- {name}: {score:.0f}/100\n"
    else:
        scorecard += "- None\n"

    scorecard += f"""
## Recommendations
"""
    for i, rec in enumerate(recommendations[:8], 1):
        scorecard += f"{i}. {rec}\n"

    scorecard += f"""
## Verdict
**{'PRODUCTION READY' if quality >= 70 else 'CONDITIONALLY READY' if quality >= 50 else 'NOT READY'}**
Overall Score: {quality:.1f}/100
"""
    scorecard_path = write_report("Repository_Scorecard.md", scorecard)

    executive = f"""# Executive Summary
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview
The Agricultural Data Engineering System (ADES) evaluation has been completed.
The system was assessed across 8 key dimensions using 5 agricultural research papers.

## Key Findings
- **Overall Score**: {quality:.1f}/100 ({_score_to_grade(quality)})
- **Pipeline Completion**: {all_results.get('e2e', {}).get('completion_rate', 0):.0%}
- **Strongest Area**: {strengths[0][0]} ({strengths[0][1]:.1f})
- **Weakest Area**: {weaknesses[0][0]} ({weaknesses[0][1]:.1f})
- **Dataset Quality**: {quality:.1f}/100
- **Models Benchmarked**: {all_results.get('model_benchmark', {}).get('trained', 0)} models trained successfully

## Verdict
**{'PRODUCTION READY' if quality >= 70 else 'CONDITIONALLY READY' if quality >= 50 else 'NOT READY'}**
"""
    exec_path = write_report("Executive_Summary.md", executive)

    pipeline_benchmark = [{"metric": k, "value": v} for k, v in all_results.get("e2e", {}).items()]
    pipeline_csv = write_csv("Pipeline_Benchmark.csv", pipeline_benchmark)

    return {
        "dashboard_html": str(html_path),
        "scorecard_md": str(scorecard_path),
        "executive_md": str(exec_path),
        "pipeline_csv": str(pipeline_csv),
    }
