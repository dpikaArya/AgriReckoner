"""
Table Intelligence Agent
Identifies and extracts structured data from agricultural research tables.
Handles Mean±SD, SEM, LSD, CD formats, replication counts, control groups,
and dose-response relationships.
"""

import re
from collections import defaultdict
from typing import Any

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings

TABLE_TYPE_PATTERNS: dict[str, list[re.Pattern]] = {
    "treatment": [
        re.compile(r"treatment|treat", re.IGNORECASE),
        re.compile(r"fertiliz|manur|amend", re.IGNORECASE),
        re.compile(r"dose|rate|application", re.IGNORECASE),
        re.compile(r"control|check|ck|t0", re.IGNORECASE),
    ],
    "yield": [
        re.compile(r"yield|productiv|grain|harvest", re.IGNORECASE),
        re.compile(r"biomass|dry.?matter|dm", re.IGNORECASE),
        re.compile(r"100.?seed|thousand.?grain|tgw", re.IGNORECASE),
        re.compile(r"harvest.?index|hi\b", re.IGNORECASE),
    ],
    "growth": [
        re.compile(r"plant.?height|height|tall", re.IGNORECASE),
        re.compile(r"leaf|leaves|la\b|leaf.?area", re.IGNORECASE),
        re.compile(r"tiller|branch|node|stem", re.IGNORECASE),
        re.compile(r"root|shoot|spike|ear", re.IGNORECASE),
        re.compile(r"spad|chlorophyll", re.IGNORECASE),
        re.compile(r"flower|fruit|pod", re.IGNORECASE),
    ],
    "soil": [
        re.compile(r"soil|pedolog", re.IGNORECASE),
        re.compile(r"ph\b|ec\b|organic", re.IGNORECASE),
        re.compile(r"nitrogen|phosphorus|potassium", re.IGNORECASE),
        re.compile(r"micronutrient|iron|zinc|manganese|copper", re.IGNORECASE),
    ],
    "weather": [
        re.compile(r"weather|climat|meteorolog", re.IGNORECASE),
        re.compile(r"temperature|rainfall|humidity", re.IGNORECASE),
        re.compile(r"growing.?degree|gdd|heat.?unit", re.IGNORECASE),
    ],
    "quality": [
        re.compile(r"quality|protein|gluten|fiber|fat", re.IGNORECASE),
        re.compile(r"ash|mineral|carbohydrate", re.IGNORECASE),
    ],
}

STATISTICAL_FORMATS = {
    "mean_sd": re.compile(r"(\d+\.?\d*)\s*[±±]\s*(\d+\.?\d*)", re.IGNORECASE),
    "mean_se": re.compile(r"(\d+\.?\d*)\s*[±±]\s*(\d+\.?\d*)\s*\(se\)", re.IGNORECASE),
    "mean_ci": re.compile(r"(\d+\.?\d*)\s*\((\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\)", re.IGNORECASE),
    "mean_lsd": re.compile(r"(\d+\.?\d*)\s*\(lsd\s*[=:]\s*(\d+\.?\d*)\)", re.IGNORECASE),
    "mean_cd": re.compile(r"(\d+\.?\d*)\s*\(cd\s*[=:]\s*(\d+\.?\d*)\)", re.IGNORECASE),
    "mean_cvar": re.compile(r"(\d+\.?\d*)\s*\(cv\s*[=:]\s*(\d+\.?\d*)%?\)", re.IGNORECASE),
    "superscript_significance": re.compile(r"(\d+\.?\d*)([a-dA-D])"),
}

CONTROL_KEYWORDS = {
    "control",
    "ctrl",
    "check",
    "ck",
    "t0",
    "untreated",
    "unfertilized",
    "water_only",
    "absolute_control",
    "negative_control",
    "blank",
}

DOSE_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(kg|g|mg|l|ml|%)\s*(/ha|/plot|/plant|/m2|/m²)?",
    re.IGNORECASE,
)


class TableIntelligenceAgent(BaseAgent):
    def __init__(self, settings: AgriAISettings | None = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self.extraction_report: dict[str, Any] = {}

    @property
    def agent_name(self) -> str:
        return "TableIntelligenceAgent"

    def process(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tables = kwargs.get("tables", [])
        raw_text = kwargs.get("raw_text", "")

        if tables:
            extracted = self._process_structured_tables(tables)
        elif raw_text:
            extracted = self._process_raw_text(raw_text)
        elif df is not None and not df.empty:
            extracted = self._process_dataframe(df)
        else:
            return df if df is not None else pd.DataFrame()

        if not extracted:
            return df

        result_df = pd.DataFrame(extracted)
        result_df = self._harmonize_schema(result_df)
        self.extraction_report = self._build_report(extracted)
        return result_df

    def _process_structured_tables(self, tables: list[dict]) -> list[dict]:
        all_rows: list[dict] = []
        for table in tables:
            table_type = self._classify_table(table)
            rows = self._extract_from_table(table, table_type)
            all_rows.extend(rows)
        return all_rows

    def _classify_table(self, table: dict) -> str:
        headers = table.get("headers", [])
        caption = table.get("caption", "")
        text = " ".join(headers) + " " + caption

        scores: dict[str, int] = {}
        for table_type, patterns in TABLE_TYPE_PATTERNS.items():
            score = sum(1 for p in patterns if p.search(text))
            if score > 0:
                scores[table_type] = score

        if not scores:
            return "unknown"
        return max(scores, key=scores.get)

    def _extract_from_table(self, table: dict, table_type: str) -> list[dict]:
        headers = table.get("headers", [])
        data_rows = table.get("rows", [])
        if not headers or not data_rows:
            return []

        treatment_col = self._find_column(headers, ["treatment", "treat", "fertilizer", "dose"])
        if treatment_col is None:
            treatment_col = 0

        extracted: list[dict] = []
        for row in data_rows:
            if not row or all(v is None or str(v).strip() == "" for v in row):
                continue

            record: dict[str, Any] = {
                "Source_File": table.get("source_file", ""),
                "Table_Type": table_type,
                "_extraction_method": "table_intelligence",
            }

            if treatment_col < len(row):
                record["Treatment"] = str(row[treatment_col]).strip()

            for i, header in enumerate(headers):
                if i >= len(row):
                    continue
                value = row[i]
                if value is None or str(value).strip() == "":
                    continue

                parsed = self._parse_statistical_value(str(value))
                if parsed:
                    record[header] = parsed["mean"]
                    if parsed.get("sd"):
                        record[f"{header}_SD"] = parsed["sd"]
                    if parsed.get("se"):
                        record[f"{header}_SE"] = parsed["se"]
                    if parsed.get("lsd"):
                        record[f"{header}_LSD"] = parsed["lsd"]
                    if parsed.get("cd"):
                        record[f"{header}_CD"] = parsed["cd"]
                    if parsed.get("cv"):
                        record[f"{header}_CV"] = parsed["cv"]
                    if parsed.get("significance"):
                        record[f"{header}_Sig"] = parsed["significance"]
                else:
                    numeric_val = self._parse_numeric(str(value))
                    if numeric_val is not None:
                        record[header] = numeric_val
                    else:
                        record[header] = str(value).strip()

            if self._is_control_treatment(record.get("Treatment", "")):
                record["Is_Control"] = True

            dose_info = self._extract_dose(record.get("Treatment", ""))
            if dose_info:
                record["Dose_Value"] = dose_info["value"]
                record["Dose_Unit"] = dose_info["unit"]
                record["Dose_Per"] = dose_info.get("per", "")

            extracted.append(record)

        return extracted

    def _process_raw_text(self, raw_text: str) -> list[dict]:
        lines = raw_text.split("\n")
        extracted: list[dict] = []
        current_table: dict[str, Any] = {"headers": [], "rows": []}

        for line in lines:
            line = line.strip()
            if not line:
                if current_table["headers"] and current_table["rows"]:
                    table_type = self._classify_table(current_table)
                    rows = self._extract_from_table(current_table, table_type)
                    extracted.extend(rows)
                current_table = {"headers": [], "rows": []}
                continue

            parts = re.split(r"\t+|\s{2,}|\|", line)
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) >= 2:
                is_header = any(
                    re.search(r"treatment|yield|height|ph|n\b|p\b|k\b", p, re.IGNORECASE)
                    for p in parts
                )
                if is_header and not current_table["headers"]:
                    current_table["headers"] = parts
                else:
                    current_table["rows"].append(parts)

        if current_table["headers"] and current_table["rows"]:
            table_type = self._classify_table(current_table)
            rows = self._extract_from_table(current_table, table_type)
            extracted.extend(rows)

        return extracted

    def _process_dataframe(self, df: pd.DataFrame) -> list[dict]:
        extracted: list[dict] = []
        for _, row in df.iterrows():
            record = row.to_dict()
            record["_extraction_method"] = "dataframe_passthrough"
            extracted.append(record)
        return extracted

    def _find_column(self, headers: list[str], keywords: list[str]) -> int | None:
        for i, header in enumerate(headers):
            header_lower = header.lower()
            for kw in keywords:
                if kw in header_lower:
                    return i
        return None

    def _parse_statistical_value(self, value: str) -> dict | None:
        value = str(value).strip()
        if not value or value in ("-", "–", "—", "ns", "NA", "N/A"):
            return None

        result: dict[str, Any] = {}

        for fmt_name, pattern in STATISTICAL_FORMATS.items():
            match = pattern.search(value)
            if match:
                groups = match.groups()
                result["mean"] = float(groups[0])

                if fmt_name == "mean_sd" and len(groups) >= 2:
                    result["sd"] = float(groups[1])
                elif fmt_name == "mean_se" and len(groups) >= 2:
                    result["se"] = float(groups[1])
                elif fmt_name == "mean_lsd" and len(groups) >= 2:
                    result["lsd"] = float(groups[1])
                elif fmt_name == "mean_cd" and len(groups) >= 2:
                    result["cd"] = float(groups[1])
                elif fmt_name == "mean_cvar" and len(groups) >= 2:
                    result["cv"] = float(groups[1])
                elif fmt_name == "mean_ci" and len(groups) >= 3:
                    result["ci_lower"] = float(groups[1])
                    result["ci_upper"] = float(groups[2])

                if fmt_name == "superscript_significance" and len(groups) >= 2:
                    result["significance"] = groups[1]

                return result

        superscript_match = STATISTICAL_FORMATS["superscript_significance"].search(value)
        if superscript_match:
            result["mean"] = float(superscript_match.group(1))
            result["significance"] = superscript_match.group(2)
            return result

        numeric = self._parse_numeric(value)
        if numeric is not None:
            result["mean"] = numeric
            return result

        return None

    def _parse_numeric(self, value: str) -> float | None:
        s = str(value).strip()
        s = re.sub(r"[†‡*]", "", s)
        s = re.sub(r"\s*±\s*.*", "", s)
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            match = re.search(r"(\d+\.?\d*)", s)
            return float(match.group(1)) if match else None

    def _is_control_treatment(self, treatment) -> bool:
        if not isinstance(treatment, str):
            return False
        t = treatment.lower().strip()
        t = re.sub(r"[†‡*]", "", t)
        return any(kw in t for kw in CONTROL_KEYWORDS)

    def _extract_dose(self, treatment) -> dict | None:
        if not isinstance(treatment, str):
            return None
        match = DOSE_PATTERN.search(treatment)
        if match:
            return {
                "value": float(match.group(1)),
                "unit": match.group(2),
                "per": match.group(3) or "",
            }
        return None

    def _harmonize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        essential_cols = [
            "Source_File",
            "Treatment",
            "Table_Type",
            "_extraction_method",
            "Is_Control",
            "Dose_Value",
            "Dose_Unit",
            "Dose_Per",
        ]
        for col in essential_cols:
            if col not in df.columns:
                df[col] = None
        return df

    def _build_report(self, extracted: list[dict]) -> dict:
        type_counts: dict[str, int] = defaultdict(int)
        stats_count = 0
        control_count = 0

        for record in extracted:
            type_counts[record.get("Table_Type", "unknown")] += 1
            if any(k.endswith(("_SD", "_SE", "_LSD", "_CD", "_CV")) for k in record):
                stats_count += 1
            if record.get("Is_Control"):
                control_count += 1

        return {
            "total_rows_extracted": len(extracted),
            "table_types": dict(type_counts),
            "rows_with_statistics": stats_count,
            "control_rows": control_count,
        }
