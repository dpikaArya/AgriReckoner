"""
Ready Reckoner Agent
Generates comparison tables, interactive dashboards, and export files
showing fertilizer scenarios with predicted outcomes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent


RECOMMENDATION_COLS = [
    "Crop", "Variety", "Location", "Soil_pH", "Nitrogen",
    "Phosphorus", "Potassium", "Rainfall", "Temperature_Max",
    "Predicted_Yield", "Expected_Biomass", "Expected_Plant_Height",
    "Recommended_Fertilizer", "Recommended_Dose",
    "Recommended_Application_Interval", "Expected_Yield_Increase",
    "Confidence_Score", "Recommendation_Summary",
]


class ReadyReckonerAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "ReadyReckonerAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        reck_dir = self.settings.RECKONER_DIR
        reck_dir.mkdir(parents=True, exist_ok=True)

        # CSV comparison table
        self._export_csv(df, reck_dir)

        # HTML comparison table
        self._export_html_table(df, reck_dir)

        # Interactive dashboard
        self._export_dashboard(df, reck_dir)

        # Report (HTML, with PDF fallback)
        self._export_report(df, reck_dir)

        self.log.info("Ready reckoner exports saved to %s", reck_dir)
        self.dataframe = df
        return df

    def _reckoner_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in RECOMMENDATION_COLS if c in df.columns]
        if not cols:
            return df.iloc[:, :6].copy()
        return df[cols].copy()

    def _export_csv(self, df: pd.DataFrame, reck_dir: Path):
        data = self._reckoner_data(df)
        path = reck_dir / "ready_reckoner.csv"
        data.to_csv(path, index=False, encoding="utf-8-sig")
        self.contract.artifacts.append(str(path))
        self.log.info("Saved: %s (%d rows)", path.name, len(data))

    def _export_html_table(self, df: pd.DataFrame, reck_dir: Path):
        data = self._reckoner_data(df)
        cols = list(data.columns)
        now = datetime.now().isoformat()

        thead = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
        tbody = ""
        for _, row in data.iterrows():
            tbody += "<tr>" + "".join(
                f"<td>{v:.2f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                for v in row
            ) + "</tr>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Ready Reckoner</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; }}
h1 {{ color: #27ae60; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #bbb; padding: 8px; text-align: left; }}
th {{ background: #27ae60; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr:hover {{ background: #e8f8f0; }}
</style>
</head>
<body>
<h1>Fertilizer Ready Reckoner</h1>
<p>Generated: {now} | {len(data)} scenarios</p>
<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>
</body>
</html>"""
        path = reck_dir / "ready_reckoner.html"
        path.write_text(html, encoding="utf-8")
        self.contract.artifacts.append(str(path))
        self.log.info("Saved: %s", path.name)

    def _export_dashboard(self, df: pd.DataFrame, reck_dir: Path):
        data = self._reckoner_data(df)
        cols = list(data.columns)
        for c in data.columns:
            data[c] = data[c].apply(
                lambda v: f"{v:.2f}" if isinstance(v, float) else v
            )

        json_data = data.to_json(orient="records")
        json_cols = json.dumps(cols)
        now = datetime.now().isoformat()

        html = self._DASHBOARD_TEMPLATE.replace("__JSON_DATA__", json_data)
        html = html.replace("__JSON_COLS__", json_cols)
        html = html.replace("__TIMESTAMP__", now)

        path = reck_dir / "dashboard.html"
        path.write_text(html, encoding="utf-8")
        self.contract.artifacts.append(str(path))
        self.log.info("Saved: %s", path.name)

    def _export_report(self, df: pd.DataFrame, reck_dir: Path):
        data = self._reckoner_data(df)
        cols = list(data.columns)
        now = datetime.now().isoformat()
        n = min(100, len(data))

        thead = "".join(f"<th>{c.replace('_', ' ')}</th>" for c in cols)
        tbody = ""
        for _, row in data.head(n).iterrows():
            tbody += "<tr>" + "".join(
                f"<td>{v:.2f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
                for v in row
            ) + "</tr>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Recommendation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; color: #2c3e50; }}
h1 {{ color: #27ae60; border-bottom: 2px solid #27ae60; padding-bottom: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 11px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
th {{ background: #27ae60; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.footer {{ margin-top: 20px; font-size: 11px; color: #95a5a6; text-align: center; }}
</style>
</head>
<body>
<h1>Recommendation Report</h1>
<p>Generated: {now} | Top {n} of {len(data)} scenarios</p>
<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>
<div class="footer">ADES Ready Reckoner</div>
</body>
</html>"""
        path = reck_dir / "report.html"
        path.write_text(html, encoding="utf-8")
        self.contract.artifacts.append(str(path))

        pdf_path = reck_dir / "report.pdf"
        try:
            import weasyprint
            weasyprint.HTML(string=html).write_pdf(pdf_path)
            self.contract.artifacts.append(str(pdf_path))
            self.log.info("Saved: %s", pdf_path.name)
        except ImportError:
            pass

    _DASHBOARD_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Recommendation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; color: #2c3e50; }
.header { background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; padding: 24px 32px; }
.header h1 { font-size: 24px; }
.header p { opacity: 0.85; margin-top: 4px; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
.card { background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 20px; margin-bottom: 20px; }
.card h2 { font-size: 16px; color: #27ae60; margin-bottom: 12px; }
.chart-row { display: flex; gap: 20px; flex-wrap: wrap; }
.chart-box { flex: 1; min-width: 300px; }
.filters { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
.filters select, .filters input { padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px; }
.filters button { padding: 8px 16px; background: #27ae60; color: white; border: none; border-radius: 6px; cursor: pointer; }
.filters button:hover { background: #219a52; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #eee; }
th { background: #f8f9fa; font-weight: 600; color: #555; cursor: pointer; position: sticky; top: 0; }
th:hover { background: #eef; }
tr:hover { background: #f0faf4; }
.highlight { background: #d4efdf !important; font-weight: 600; }
.stats { display: flex; gap: 16px; flex-wrap: wrap; }
.stat { background: white; border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 140px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.stat .value { font-size: 24px; font-weight: 700; color: #27ae60; }
.stat .label { font-size: 12px; color: #7f8c8d; margin-top: 2px; }
.footer { text-align: center; padding: 20px; color: #95a5a6; font-size: 12px; }
</style>
</head>
<body>
<div class="header"><h1>Fertilizer Recommendation Dashboard</h1><p>Generated: __TIMESTAMP__</p></div>
<div class="container">
<div class="stats" id="stats"></div>
<div class="card"><h2>Yield by Fertilizer Scenario</h2>
<div class="chart-row"><div class="chart-box"><canvas id="scatterChart"></canvas></div></div></div>
<div class="card"><h2>Recommendations</h2>
<div class="filters">
<input type="text" id="filterInput" placeholder="Search..." onkeyup="filterTable()">
<select id="sortSelect" onchange="renderTable()"><option value="">Sort by...</option></select>
<button onclick="renderTable()">Refresh</button>
</div>
<div style="overflow-x:auto;" id="tableContainer"></div>
</div></div>
<div class="footer">ADES Ready Reckoner</div>
<script>
const DATA = __JSON_DATA__;
const COLS = __JSON_COLS__;
function renderStats() {
  const el = document.getElementById('stats'); const n = DATA.length;
  const avgYield = DATA.reduce((s,r) => s + parseFloat(r.Predicted_Yield||0), 0) / n || 0;
  const avgConf = DATA.reduce((s,r) => s + parseFloat(r.Confidence_Score||0), 0) / n || 0;
  const recs = new Set(DATA.map(r => r.Recommended_Fertilizer).filter(Boolean)).size;
  el.innerHTML = '<div class="stat"><div class="value">'+n+'</div><div class="label">Scenarios</div></div>' +
    '<div class="stat"><div class="value">'+avgYield.toFixed(2)+'</div><div class="label">Avg Yield</div></div>' +
    '<div class="stat"><div class="value">'+(avgConf*100).toFixed(0)+'%</div><div class="label">Avg Confidence</div></div>' +
    '<div class="stat"><div class="value">'+recs+'</div><div class="label">Fertilizers</div></div>';
}
function renderChart() {
  new Chart(document.getElementById('scatterChart'), {
    type: 'bar',
    data: { labels: DATA.map(r => r.Recommended_Fertilizer || 'N/A'),
      datasets: [{ label: 'Predicted Yield', data: DATA.map(r => parseFloat(r.Predicted_Yield)||0), backgroundColor: '#27ae60' }] },
    options: { responsive: true, scales: { y: { beginAtZero: true, title: { display: true, text: 'Yield' } } } }
  });
}
function renderTable() {
  const sortKey = document.getElementById('sortSelect').value;
  const filterVal = document.getElementById('filterInput').value.toLowerCase();
  let data = [...DATA];
  if (filterVal) data = data.filter(r => JSON.stringify(r).toLowerCase().includes(filterVal));
  if (sortKey) data.sort((a,b) => parseFloat(b[sortKey]||0) - parseFloat(a[sortKey]||0));
  const best = data.length ? Math.max(...data.map(r => parseFloat(r.Predicted_Yield)||0)) : 0;
  let html = '<table><thead><tr>'+COLS.map(c => '<th onclick="sortBy(\\''+c+'\\')">'+c.replace(/_/g,' ')+'</th>').join('')+'</tr></thead><tbody>';
  data.forEach(r => { const ib = parseFloat(r.Predicted_Yield) === best;
    html += '<tr class="'+(ib?'highlight':'')+'">'+COLS.map(c => '<td>'+(r[c]||'')+'</td>').join('')+'</tr>';
  });
  document.getElementById('tableContainer').innerHTML = html+'</tbody></table>';
}
function sortBy(key) { document.getElementById('sortSelect').value = key; renderTable(); }
const sortSel = document.getElementById('sortSelect');
COLS.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c.replace(/_/g,' '); sortSel.appendChild(o); });
renderStats(); renderChart(); renderTable();
</script>
</body>
</html>"""

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "exports_complete": True,
        }
