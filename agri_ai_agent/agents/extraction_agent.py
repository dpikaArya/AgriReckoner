"""
Consolidated Extraction Agent.
Flow: Read file/paper → Extract structured data → Map to ontology → Convert to UAMS schema.
"""

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from agri_ai_agent.agents.base_agent import BaseAgent
from agri_ai_agent.config.settings import AgriAISettings
from agri_ai_agent.config.schema import UAMS_COLUMNS, VARIANT_MAP

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

KNOWN_CROPS = {
    "bell pepper", "capsicum annuum", "capsicum", "black wheat", "triticum aestivum",
    "carrot", "daucus carota", "cowpea", "vigna unguiculata", "spinach", "spinacia oleracea",
    "rice", "oryza sativa", "wheat", "triticum", "maize", "zea mays", "corn",
    "soybean", "glycine max", "potato", "solanum tuberosum", "tomato", "solanum lycopersicum",
    "chickpea", "cicer arietinum", "pigeonpea", "cajanus cajan", "groundnut", "arachis hypogaea",
    "mustard", "brassica juncea", "sunflower", "helianthus annuus", "sugarcane", "saccharum",
    "cotton", "gossypium", "onion", "allium cepa", "chilli", "capsicum frutescens",
    "bengal gram", "green gram", "black gram", "cluster bean", "horse gram",
    "fenugreek", "cumin", "coriander", "fennel", "sesame", "linseed", "castor",
    "jute", "tea", "coffee", "finger millet", "pearl millet", "foxtail millet",
    "sorghum", "millet", "barley", "oat", "lentil", "peas", "cabbage", "cauliflower",
    "broccoli", "brinjal", "eggplant", "cucumber", "pumpkin", "watermelon", "muskmelon",
    "banana", "mango", "orange", "grape", "apple", "strawberry",
}

DESIGN_KEYWORDS = [
    "randomized block design", "rcbd", "randomized complete block design",
    "split plot", "split-plot", "crd", "completely randomized design",
    "latin square", "factorial", "nested design", "strip plot",
]

ENTITY_TYPES = {
    "Crop": KNOWN_CROPS,
    "Variety": set(),
    "Fertilizer": {"urea", "dap", "npk", "compost", "vermicompost", "farmyard manure",
                    "fym", "potassium", "phosphorus", "nitrogen", "biofertilizer",
                    "rhizobium", "azotobacter", "psb", "pgpr", "organic fertilizer"},
    "Growth_Parameter": {"plant height", "shoot length", "root length", "leaf area",
                          "leaf number", "branches", "flowers", "fruits", "spike length",
                          "panicle length", "pod length", "seed weight", "test weight",
                          "biomass", "dry weight", "fresh weight"},
    "Yield": {"yield", "grain yield", "seed yield", "fruit yield", "dry yield",
               "biological yield", "harvest index"},
    "Quality_Trait": {"protein", "protein content", "oil content", "fiber", "ash",
                       "carbohydrate", "starch", "sugar", "vitamin", "mineral",
                       "iron", "zinc", "calcium", "magnesium", "selenium"},
    "Soil_Property": {"ph", "soil ph", "ec", "electrical conductivity", "organic carbon",
                       "oc", "available nitrogen", "available phosphorus", "available potassium",
                       "nitrogen", "phosphorus", "potassium", "sulfur", "zinc", "iron",
                       "manganese", "copper", "boron"},
    "Weather": {"temperature", "rainfall", "humidity", "solar radiation", "wind speed",
                 "evapotranspiration", "max temperature", "min temperature", "tmax", "tmin"},
}

RANGE_CONSTRAINTS = {
    "soil_ph": (3.0, 10.0), "ec_ds_m": (0.0, 10.0), "organic_carbon_pct": (0.0, 10.0),
    "nitrogen_kg_ha": (0.0, 1000.0), "phosphorus_kg_ha": (0.0, 500.0),
    "potassium_kg_ha": (0.0, 1000.0), "temperature_max_c": (-20.0, 55.0),
    "temperature_min_c": (-30.0, 40.0), "rainfall_mm": (0.0, 10000.0),
    "humidity_pct": (0.0, 100.0), "yield_kg_ha": (0.0, 50000.0),
    "plant_height_cm": (0.0, 500.0), "spad": (0.0, 80.0), "protein_pct": (0.0, 60.0),
    "fruit_weight_g": (0.0, 5000.0), "fruit_number": (0.0, 10000.0),
    "harvest_index": (0.0, 1.5), "100_seed_weight_g": (0.0, 500.0),
}

ONTOLOGY_KNOWLEDGE_BASE = {
    "Crop": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_2556",
        "CropOntology": "CO_320:0000000", "FoodOn": "FOODON:00001002",
        "label": "Crop", "synonyms": ["crop type", "cultivated plant", "crop species"],
        "parent": "Agricultural product",
        "children": ["Cereal crop", "Vegetable crop", "Fruit crop", "Legume crop"],
    },
    "Variety": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_8151",
        "CropOntology": "CO_320:0000001", "label": "Variety",
        "synonyms": ["cultivar", "variety", "variety type", "variety name"],
        "parent": "Crop", "children": [],
    },
    "Soil_pH": {
        "ENVO": "http://purl.obolibrary.org/obo/ENVO_00001995",
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_7189",
        "label": "Soil pH", "synonyms": ["pH", "soil acidity", "soil reaction", "pH value"],
        "parent": "Soil property", "children": [],
    },
    "EC": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_2482",
        "label": "Electrical conductivity",
        "synonyms": ["EC", "electrical conductivity", "soil EC", "salinity"],
        "parent": "Soil property", "children": [],
    },
    "Organic_Carbon": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_5385",
        "ENVO": "http://purl.obolibrary.org/obo/ENVO_00002281",
        "label": "Soil organic carbon",
        "synonyms": ["OC", "organic carbon", "soil organic carbon", "total organic carbon"],
        "parent": "Soil property", "children": [],
    },
    "Nitrogen": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_5188",
        "CropOntology": "CO_320:0000020", "FoodOn": "FOODON:00001310",
        "label": "Nitrogen", "synonyms": ["N", "available nitrogen", "total nitrogen", "soil nitrogen"],
        "parent": "Plant nutrient", "children": ["Ammonium nitrogen", "Nitrate nitrogen"],
    },
    "Phosphorus": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_5801",
        "label": "Phosphorus", "synonyms": ["P", "phosphorous", "available phosphorus", "phosphate"],
        "parent": "Plant nutrient", "children": [],
    },
    "Potassium": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_6139",
        "label": "Potassium", "synonyms": ["K", "available potassium", "potash"],
        "parent": "Plant nutrient", "children": [],
    },
    "Temperature_Max": {
        "ENVO": "http://purl.obolibrary.org/obo/ENVO_01000244",
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_7651",
        "label": "Maximum temperature",
        "synonyms": ["tmax", "max temp", "maximum temperature", "daily maximum temperature"],
        "parent": "Temperature", "children": [],
    },
    "Temperature_Min": {
        "ENVO": "http://purl.obolibrary.org/obo/ENVO_01000243",
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_7652",
        "label": "Minimum temperature",
        "synonyms": ["tmin", "min temp", "minimum temperature", "daily minimum temperature"],
        "parent": "Temperature", "children": [],
    },
    "Rainfall": {
        "ENVO": "http://purl.obolibrary.org/obo/ENVO_01000507",
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_6435",
        "label": "Rainfall",
        "synonyms": ["precipitation", "rain", "rainfall amount", "precipitation amount"],
        "parent": "Weather phenomenon", "children": [],
    },
    "Humidity": {
        "ENVO": "http://purl.obolibrary.org/obo/ENVO_01000245",
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_3689",
        "label": "Humidity",
        "synonyms": ["relative humidity", "air humidity", "atmospheric humidity"],
        "parent": "Weather phenomenon", "children": [],
    },
    "Yield_per_Hectare": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_8496",
        "label": "Yield per hectare",
        "synonyms": ["yield ha", "yield per ha", "grain yield", "crop yield"],
        "parent": "Yield", "children": [],
    },
    "Plant_Height_cm": {
        "CropOntology": "CO_320:0000005",
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_330960",
        "label": "Plant height",
        "synonyms": ["height", "plant height", "shoot length"],
        "parent": "Growth parameter", "children": [],
    },
    "SPAD": {
        "CropOntology": "CO_320:0000035",
        "label": "SPAD chlorophyll",
        "synonyms": ["chlorophyll", "chlorophyll content", "SPAD value", "leaf greenness"],
        "parent": "Plant measurement", "children": [],
    },
    "Protein": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_6252",
        "FoodOn": "FOODON:00001024",
        "label": "Protein content",
        "synonyms": ["crude protein", "protein content", "grain protein"],
        "parent": "Quality parameter", "children": [],
    },
    "Harvest_Index": {
        "AGROVOC": "http://aims.fao.org/aos/agrovoc/c_8483",
        "label": "Harvest index",
        "synonyms": ["HI", "harvest index"],
        "parent": "Yield parameter", "children": [],
    },
}

ADDITIONAL_SYNONYMS = {
    "seasonal": "Season", "cropping_season": "Season", "kharif": "Season",
    "rabi": "Season", "zaid": "Season", "sowing_date": "Season", "planting_date": "Season",
    "harvest_date": "Growth_Duration_Days", "days_to_harvest": "Growth_Duration_Days",
    "crop_duration": "Growth_Duration_Days", "maturity_days": "Growth_Duration_Days",
    "no_of_replications": "Replications", "rep": "Replications", "repititions": "Replications",
    "plot_area_m2": "Plot_Size", "plot_dimension": "Plot_Size",
    "row_to_row": "Spacing_Row", "plant_to_plant": "Spacing_Plant",
    "intra_row": "Spacing_Plant", "inter_row": "Spacing_Row",
    "biofertilizers": "Biofertilizer", "bio_fertilizers": "Biofertilizer",
    "inoculation": "Biofertilizer", "organic_amendments": "Organic_Fertilizer",
    "farmyard_manure": "Organic_Fertilizer", "fym": "Organic_Fertilizer",
    "compost": "Organic_Fertilizer", "vermicompost": "Organic_Fertilizer",
    "n_dose": "Dose", "p_dose": "Dose", "k_dose": "Dose", "fertilizer_dose": "Dose",
    "n_kg_ha": "Nitrogen", "p_kg_ha": "Phosphorus", "k_kg_ha": "Potassium",
    "available_n_kg_ha": "Nitrogen", "available_p_kg_ha": "Phosphorus", "available_k_kg_ha": "Potassium",
    "total_n": "Nitrogen", "total_p": "Phosphorus", "total_k": "Potassium",
    "plant_height_at_harvest": "Plant_Height_cm", "final_plant_height": "Plant_Height_cm",
    "shoot_length_at_harvest": "Shoot_Length_cm", "root_length_at_harvest": "Root_Length_cm",
    "leaf_area_index": "Leaf_Area_cm2", "lai": "Leaf_Area_cm2",
    "chlorophyll_spad": "SPAD", "chlorophyll_content_spad": "SPAD",
    "number_of_branches": "Branches", "branches_per_plant": "Branches",
    "number_of_flowers": "Flowers", "flowers_per_plant": "Flowers",
    "number_of_nodes": "Nodes", "seed_yield": "Yield_per_Plot", "grain_yield": "Yield_per_Plot",
    "economic_yield": "Yield_per_Plot", "biological_yield": "Biomass_Yield",
    "total_biomass": "Biomass_Yield", "straw_yield": "Biomass_Yield",
    "test_weight_g": "100_Seed_Weight", "thousand_grain_weight": "100_Seed_Weight",
    "1000_grain_weight": "100_Seed_Weight",
}

# ---------------------------------------------------------------------------
# EXTRACTION AGENT
# ---------------------------------------------------------------------------


class ExtractionAgent(BaseAgent):
    def __init__(self, settings: Optional[AgriAISettings] = None, **kwargs):
        super().__init__(settings or AgriAISettings(), **kwargs)

    @property
    def agent_name(self) -> str:
        return "ExtractionAgent"

    def process(self, df: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        filepath = kwargs.get("filepath")
        papers_dir = kwargs.get("papers_dir")

        ingestion_result = None
        document_structures = []
        extracted_entities = []
        ontology_mappings = []

        # --- Step 1: Read / Ingest ---
        if df is not None:
            self.dataframe = df
            self.log.info("[%s] Using provided DataFrame with %d columns", self.agent_name, len(df.columns))
        elif filepath:
            self.dataframe, ingestion_result = self._ingest(filepath, kwargs.get("sheet_name"))
        elif papers_dir:
            self.dataframe = pd.DataFrame()
        else:
            raise ValueError("One of df, filepath, or papers_dir must be provided")

        # --- Step 2: Extract from PDFs (if papers_dir given) ---
        if papers_dir:
            extracted_entities = self._extract_from_papers(papers_dir)

        # If ingestion populated the dataframe, also extract entities from text columns
        if self.dataframe is not None and not self.dataframe.empty:
            text_cols = [c for c in self.dataframe.columns if self.dataframe[c].dtype == object]
            for col in text_cols:
                for val in self.dataframe[col].dropna().astype(str):
                    extracted_entities.extend(self._ner_from_text(val))

        # --- Step 3: Map to ontology ---
        if self.dataframe is not None and not self.dataframe.empty:
            ontology_mappings = self._map_to_ontology(self.dataframe.columns)
        elif extracted_entities:
            var_names = list({e.get("canonical_variable", e.get("variable", ""))
                              for e in extracted_entities if e.get("variable")})
            ontology_mappings = self._map_to_ontology(var_names)

        # --- Step 4: Map to UAMS schema ---
        if self.dataframe is not None and not self.dataframe.empty:
            self.dataframe = self._map_to_schema(self.dataframe)

        # Save artifacts
        self._save_artifacts(ingestion_result, document_structures, extracted_entities, ontology_mappings)

        return self.dataframe if self.dataframe is not None else pd.DataFrame()

    # ------------------------------------------------------------------
    # STEP 1: Ingestion
    # ------------------------------------------------------------------

    def _ingest(self, filepath: str, sheet_name: Optional[str] = None) -> tuple[pd.DataFrame, dict]:
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")

        ext = fp.suffix.lower()
        if ext == ".csv":
            enc = self._detect_encoding(fp)
            df = pd.read_csv(fp, encoding=enc)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(fp, sheet_name=sheet_name)
        elif ext == ".json":
            df = pd.read_json(fp)
        elif ext == ".parquet":
            df = pd.read_parquet(fp)
        else:
            df = pd.read_csv(fp)

        result = {
            "file": fp.name,
            "rows": len(df),
            "columns": len(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        self.log.info("[%s] Ingested %s: %d rows x %d cols", self.agent_name, fp.name, len(df), len(df.columns))
        return df, result

    def _detect_encoding(self, fp: Path) -> str:
        try:
            import chardet
            raw = fp.read_bytes()
            return chardet.detect(raw)["encoding"] or "utf-8"
        except ImportError:
            return "utf-8"

    # ------------------------------------------------------------------
    # STEP 2: PDF Document Understanding & Scientific Extraction
    # ------------------------------------------------------------------

    def _extract_from_papers(self, papers_dir: str) -> list[dict]:
        pdf_dir = Path(papers_dir)
        pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.is_dir() else [pdf_dir]
        all_entities = []

        for pdf_path in pdf_files:
            if not pdf_path.exists():
                continue
            self.log.info("[%s] Processing paper: %s", self.agent_name, pdf_path.name)

            doc = self._parse_document(pdf_path)
            text = doc.get("full_text", "")
            if not text:
                self.log.warning("[%s] No text in %s", self.agent_name, pdf_path.name)
                continue

            entities = self._ner_from_text(text, source=pdf_path.name)
            all_entities.extend(entities)

            self.log.info("[%s] Extracted %d entities from %s", self.agent_name, len(entities), pdf_path.name)

        return all_entities

    def _parse_document(self, pdf_path: Path) -> dict:
        text = self._extract_text(pdf_path)
        lines = text.split("\n") if text else []

        title = ""
        for i, line in enumerate(lines[:5]):
            s = line.strip()
            if len(s) > 20:
                title = s
                break

        doi_match = re.search(r"(10\.\d{4,}/[-._;()/:\w\d]+)", text)
        doi = doi_match.group(1) if doi_match else ""

        sections = defaultdict(int)
        section_names = {"abstract", "introduction", "materials and methods", "results",
                         "discussion", "conclusion", "references", "acknowledgement"}
        for line in lines:
            s = line.strip().lower()
            if s in section_names:
                sections[s] += 1

        ref_count = 0
        in_refs = False
        for line in lines:
            s = line.strip()
            if re.match(r"^\s*(references|bibliography)\s*$", s, re.IGNORECASE):
                in_refs = True
                continue
            if in_refs and len(s) > 15:
                ref_count += 1

        return {
            "filename": pdf_path.name,
            "doi": doi,
            "full_text": text,
            "title": title,
            "page_count": max(1, len(lines) // 50),
            "word_count": len(text.split()),
            "char_count": len(text),
            "sections": dict(sections),
            "reference_count": ref_count,
        }

    def _extract_text(self, pdf_path: Path) -> str:
        try:
            from pdfminer.high_level import extract_text
            return extract_text(str(pdf_path))
        except ImportError:
            pass
        try:
            import PyPDF2
            parts = []
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        except ImportError:
            pass
        try:
            result = subprocess.run(
                ["pdftotext", str(pdf_path), "-"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            self.log.warning("[%s] pdftotext extraction failed for %s: %s",
                             self.agent_name, pdf_path.name, e)
        return ""

    def _ner_from_text(self, text: str, source: str = "text") -> list[dict]:
        entities = []
        seen = set()

        def add(etype: str, value: str, var: str = "", page: int = 1):
            key = f"{etype}:{value.lower().strip()}"
            if key in seen:
                return
            seen.add(key)
            entities.append({
                "entity_type": etype,
                "value": value.strip(),
                "variable": var or value.strip(),
                "canonical_variable": "",
                "source": source,
                "page": page,
                "confidence": "medium",
            })

        page = 1

        for crop in KNOWN_CROPS:
            for m in re.finditer(re.escape(crop), text, re.IGNORECASE):
                add("Crop", m.group(0), page=page)

        for design in DESIGN_KEYWORDS:
            for m in re.finditer(re.escape(design), text, re.IGNORECASE):
                add("Experimental_Design", m.group(0), page=page)

        for m in re.finditer(r"(\d+)\s*(?:replications?|replicates?|reps?)\b", text, re.IGNORECASE):
            add("Replications", m.group(0), page=page)

        patterns = [
            (r"(?:max\s*temp|tmax|temperature\s*max)\s*[:=]?\s*([\d.]+)\s*°?\s*C",
             "Weather", "Temperature_Max"),
            (r"(?:min\s*temp|tmin|temperature\s*min)\s*[:=]?\s*([\d.]+)\s*°?\s*C",
             "Weather", "Temperature_Min"),
            (r"(?:rainfall|precipitation)\s*[:=]?\s*([\d.]+)\s*(mm)", "Weather", "Rainfall"),
            (r"(?:yield|grain\s*yield|seed\s*yield)\s*[:=]?\s*([\d.]+)\s*(kg/ha|t/ha|q/ha)",
             "Yield", "Yield_kg_ha"),
            (r"(?:plant\s*height|height|shoot\s*length)\s*[:=]?\s*([\d.]+)\s*cm",
             "Growth_Parameter", "Plant_Height_cm"),
            (r"(?:spad|chlorophyll)\s*[:=]?\s*([\d.]+)", "Quality_Trait", "SPAD"),
            (r"(?:protein|protein\s*content)\s*[:=]?\s*([\d.]+)\s*%", "Quality_Trait", "Protein_pct"),
            (r"(?:soil\s*ph|ph)\s*[:=]?\s*([\d.]+)", "Soil_Property", "Soil_pH"),
            (r"(?:ec|electrical\s*conductivity)\s*[:=]?\s*([\d.]+)\s*(dS/m)?",
             "Soil_Property", "EC"),
            (r"(?:organic\s*carbon|oc)\s*[:=]?\s*([\d.]+)\s*%?", "Soil_Property", "Organic_Carbon_pct"),
        ]

        for pattern, etype, var in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                add(etype, m.group(0), var=var, page=page)

        return entities

    # ------------------------------------------------------------------
    # STEP 3: Ontology Mapping
    # ------------------------------------------------------------------

    def _map_to_ontology(self, variables) -> list[dict]:
        rows = []
        for col in variables:
            base = col
            for suffix in ["_cm", "_g", "_mm", "_cm2", "_m2", "_kg", "_mg",
                           "_ds_m", "_ppm", "_30_cm", "_60_cm", "_90_cm",
                           "_Code", "_Calc", "_7d_MA"]:
                if col.endswith(suffix):
                    base = col[:-len(suffix)]
                    break

            entry = ONTOLOGY_KNOWLEDGE_BASE.get(col) or ONTOLOGY_KNOWLEDGE_BASE.get(base)
            if entry:
                rows.append({
                    "Variable": col,
                    "Preferred_Label": entry.get("label", base),
                    "AGROVOC_ID": entry.get("AGROVOC", ""),
                    "CropOntology_ID": entry.get("CropOntology", ""),
                    "ENVO_ID": entry.get("ENVO", ""),
                    "FoodOn_ID": entry.get("FoodOn", ""),
                    "Synonyms": "; ".join(entry.get("synonyms", [])),
                    "Parent_Concept": entry.get("parent", ""),
                })
            else:
                rows.append({
                    "Variable": col, "Preferred_Label": col,
                    "AGROVOC_ID": "", "CropOntology_ID": "",
                    "ENVO_ID": "", "FoodOn_ID": "",
                    "Synonyms": "", "Parent_Concept": "",
                })

        mapped = sum(1 for r in rows if r["AGROVOC_ID"] or r["CropOntology_ID"] or r["ENVO_ID"] or r["FoodOn_ID"])
        self.log.info("[%s] Ontology mapped %d/%d variables", self.agent_name, mapped, len(rows))
        return rows

    # ------------------------------------------------------------------
    # STEP 4: UAMS Schema Mapping
    # ------------------------------------------------------------------

    def _map_to_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {}
        unmapped = []

        for col in df.columns:
            norm = self._normalize(col)
            target = self._resolve_column(norm)
            if target:
                rename_map[col] = target
            else:
                unmapped.append(col)
                rename_map[col] = col.strip()

        result_df = df.rename(columns=rename_map, errors="ignore")
        result_df = self._resolve_duplicate_columns(result_df)

        for col in UAMS_COLUMNS:
            if col not in result_df.columns:
                result_df[col] = pd.NA

        ordered = [c for c in UAMS_COLUMNS if c in result_df.columns]
        extra = [c for c in result_df.columns if c not in UAMS_COLUMNS]
        result_df = result_df[ordered + extra]

        if unmapped:
            self.log.warning("[%s] %d columns unmapped: %s", self.agent_name, len(unmapped), unmapped[:10])

        self.log.info("[%s] Schema mapped: %d cols → %d UAMS cols",
                      self.agent_name, len(df.columns), len(ordered))
        return result_df

    def _resolve_column(self, norm: str) -> str:
        result = self._match_patterns(norm)
        if result:
            return result
        result = VARIANT_MAP.get(norm, "")
        if result:
            return result
        result = ADDITIONAL_SYNONYMS.get(norm, "")
        return result

    def _match_patterns(self, norm: str) -> str:
        m = re.match(r"plantheight[_\s]*(\d+)[_\s]*cm", norm)
        if m:
            return f"Plant_Height_{m.group(1)}_cm"
        m = re.match(r"leaf[_\s]*area[_\s]*(\d+)[_\s]*cm2", norm)
        if m:
            return f"Leaf_Area_{m.group(1)}_cm2"
        if re.match(r"yieldplot[_\s]*\d+[_\s]*g", norm):
            return "Yield_per_Plot"
        if re.match(r"fruitweight[_\s]*\d+[_\s]*g", norm):
            return "Fruit_Weight"
        if re.match(r"fruitdiameter[_\s]*\d+[_\s]*mm", norm):
            return "Fruit_Diameter_mm"
        if re.match(r"branches[_\s]*\d+", norm):
            return "Branches"
        if re.match(r"flowers[_\s]*\d+", norm):
            return "Flowers"
        return ""

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def _validate_values(self, extractions: list[dict]) -> list[dict]:
        validated = []
        for ext in extractions:
            var = ext.get("canonical_variable", ext.get("variable", ""))
            val = ext.get("value", "")
            constraints = RANGE_CONSTRAINTS.get(var)
            if constraints and val:
                try:
                    num = float(re.search(r"[\d.]+", str(val)).group())
                    lo, hi = constraints
                    if num < lo or num > hi:
                        ext["confidence"] = "low"
                        ext["validation_note"] = f"Value {num} outside range [{lo}, {hi}]"
                except (ValueError, AttributeError, TypeError):
                    pass
            validated.append(ext)
        return validated

    # ------------------------------------------------------------------
    # ARTIFACTS
    # ------------------------------------------------------------------

    def _save_artifacts(self, ingestion: Optional[dict], structures: list[dict],
                        entities: list[dict], ontology_rows: list[dict]):
        if ingestion:
            path = self.save_text_artifact(json.dumps(ingestion, indent=2),
                                           "ingestion_report.json")
            self.log.info("[%s] Ingestion report → %s", self.agent_name, path)

        if entities:
            path = self.save_text_artifact(json.dumps(entities, indent=2, default=str),
                                           "extracted_entities.json")
            self.log.info("[%s] Entities → %s", self.agent_name, path)

        if ontology_rows:
            onto_df = pd.DataFrame(ontology_rows)
            path = self.save_artifact(onto_df, "ontology_mapping.csv")
            self.log.info("[%s] Ontology mapping → %s", self.agent_name, path)

    def _build_output(self, df: pd.DataFrame, **kwargs) -> dict:
        base = super()._build_output(df, **kwargs)
        base["schema_version"] = "1.0"
        return base
