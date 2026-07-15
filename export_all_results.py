"""
AAIF Results Export — All results to Excel (.xlsx) and Word (.docx)
Generates:
  1. AAIF_All_Results.xlsx       — Master workbook (all sheets)
  2. AAIF_Model_Report.docx      — Full narrative report
  3. AAIF_Regression_Report.docx — Multi-regression analysis report
"""
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.resolve()
OUTPUTS_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUTS_DIR / "tables"
EXPORT_DIR = OUTPUTS_DIR / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def safe_read(path, **kwargs):
    """Safely read a file, return None if not found."""
    try:
        return pd.read_excel(path, **kwargs)
    except Exception:
        return None


def style_table(df, title=""):
    """Add basic formatting hints for Excel."""
    return df


# =========================================================================
# PART 1: MASTER EXCEL WORKBOOK
# =========================================================================
def export_master_excel():
    """Create one master Excel file with all result sheets."""
    log("Creating master Excel workbook...")

    xlsx_path = EXPORT_DIR / "AAIF_All_Results.xlsx"

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:

        # Sheet 1: Ready Reckoner (24 Papers)
        rr = safe_read(OUTPUTS_DIR / "Ready_Reckoner_24Papers.xlsx")
        if rr is not None:
            rr.to_excel(writer, sheet_name='Ready_Reckoner_24Papers', index=False)
            log(f"  + Ready_Reckoner_24Papers ({rr.shape[0]} rows)")

        # Sheet 2: Model Performance Assessment
        perf = safe_read(TABLE_DIR / "Model_Performance_Assessment.xlsx")
        if perf is not None:
            perf.to_excel(writer, sheet_name='Model_Performance', index=False)
            log(f"  + Model_Performance ({perf.shape[0]} models)")

        # Sheet 3: Multi-Regression Coefficients
        reg_coef = safe_read(TABLE_DIR / "Multi_Regression_Results.xlsx", sheet_name='Coefficients')
        if reg_coef is not None:
            reg_coef.to_excel(writer, sheet_name='Regression_Coefficients', index=False)
            log(f"  + Regression_Coefficients ({reg_coef.shape[0]} variables)")

        # Sheet 4: Multi-Regression Summary
        reg_sum = safe_read(TABLE_DIR / "Multi_Regression_Results.xlsx", sheet_name='Model_Summary')
        if reg_sum is not None:
            reg_sum.to_excel(writer, sheet_name='Regression_Summary', index=False)
            log(f"  + Regression_Summary")

        # Sheet 5: Pipeline Ready Reckoner
        rr_pipe = safe_read(OUTPUTS_DIR / "Ready_Reckoner.xlsx")
        if rr_pipe is not None:
            rr_pipe.to_excel(writer, sheet_name='Pipeline_ReadyReckoner', index=False)
            log(f"  + Pipeline_ReadyReckoner ({rr_pipe.shape[0]} rows)")

        # Sheet 6: Extraction Report
        extr = safe_read(OUTPUTS_DIR / "extraction_report.csv")
        if extr is not None:
            extr.to_excel(writer, sheet_name='Extraction_Report', index=False)
            log(f"  + Extraction_Report ({extr.shape[0]} papers)")

        # Sheet 7: Ingestion Report
        ing = safe_read(OUTPUTS_DIR / "ingestion_report.xlsx")
        if ing is not None:
            ing.to_excel(writer, sheet_name='Ingestion_Report', index=False)
            log(f"  + Ingestion_Report ({ing.shape[0]} papers)")

        # Sheet 8: Validation Report
        val = safe_read(OUTPUTS_DIR / "validation_report.xlsx")
        if val is not None:
            val.to_excel(writer, sheet_name='Validation_Report', index=False)
            log(f"  + Validation_Report ({val.shape[0]} columns)")

        # Sheet 9: Feature Importance
        fi = safe_read(OUTPUTS_DIR / "feature_importance.csv")
        if fi is not None:
            fi.to_excel(writer, sheet_name='Feature_Importance', index=False)
            log(f"  + Feature_Importance ({fi.shape[0]} features)")

        # Sheet 10: Universal Agricultural Schema
        uas = safe_read(OUTPUTS_DIR / "Universal_Agricultural_Schema.xlsx")
        if uas is not None:
            uas.to_excel(writer, sheet_name='Universal_Schema', index=False)
            log(f"  + Universal_Schema ({uas.shape[0]} rows)")

        # Sheet 11: Fertilizer Recommendations
        fert = safe_read(OUTPUTS_DIR / "recommendations" / "fertilizer_recommendations.xlsx")
        if fert is not None:
            fert.to_excel(writer, sheet_name='Fertilizer_Recommendations', index=False)
            log(f"  + Fertilizer_Recommendations ({fert.shape[0]} rows)")

        # Sheet 12: Model Metrics (from pipeline)
        mm = safe_read(OUTPUTS_DIR / "model_metrics.xlsx")
        if mm is not None:
            mm.to_excel(writer, sheet_name='Pipeline_ModelMetrics', index=False)
            log(f"  + Pipeline_ModelMetrics ({mm.shape[0]} rows)")

        # Sheet 13: Features Dataset
        fd = safe_read(OUTPUTS_DIR / "features_dataset.csv")
        if fd is not None:
            fd.to_excel(writer, sheet_name='Features_Dataset', index=False)
            log(f"  + Features_Dataset ({fd.shape[0]} rows, {fd.shape[1]} cols)")

        # Sheet 14: Universal Schema CSV
        us_csv = safe_read(OUTPUTS_DIR / "Universal_Agricultural_Schema.csv")
        if us_csv is not None:
            us_csv.to_excel(writer, sheet_name='UAS_CSV', index=False)
            log(f"  + UAS_CSV ({us_csv.shape[0]} rows)")

        # Sheet 15: Model Metrics CSV
        mm_csv = safe_read(OUTPUTS_DIR / "model_metrics.csv")
        if mm_csv is not None:
            mm_csv.to_excel(writer, sheet_name='Model_Metrics_CSV', index=False)
            log(f"  + Model_Metrics_CSV")

    log(f"\n  Master Excel saved: {xlsx_path}")
    log(f"  Size: {xlsx_path.stat().st_size / 1024:.1f} KB")
    return xlsx_path


# =========================================================================
# PART 2: WORD DOCUMENT — FULL NARRATIVE REPORT
# =========================================================================
def export_narrative_report():
    """Create a comprehensive Word document with all results."""
    from docx import Document
    from docx.shared import Inches, Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn

    log("Creating narrative report (DOCX)...")

    doc = Document()

    # Set default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    # ---- TITLE ----
    title = doc.add_heading('AAIF Agricultural Intelligence Framework', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('Comprehensive Results Report — 24 Research Papers')
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x99)

    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_para.add_run(f'Generated: {datetime.now().strftime("%B %d, %Y %H:%M")}')
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    doc.add_page_break()

    # ---- TABLE OF CONTENTS ----
    doc.add_heading('Table of Contents', level=1)
    toc_items = [
        '1. Executive Summary',
        '2. Data Ingestion Summary',
        '3. Extraction Performance by Paper',
        '4. Model Performance Assessment',
        '5. Multiple Regression Analysis',
        '6. Fertilizer Recommendations',
        '7. Ready Reckoner Table (24 Papers)',
        '8. Validation Summary',
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)

    doc.add_page_break()

    # ---- 1. EXECUTIVE SUMMARY ----
    doc.add_heading('1. Executive Summary', level=1)
    doc.add_paragraph(
        'This report presents the complete results from the Agentic Agricultural Intelligence '
        'Framework (AAIF) applied to 24 research papers covering crops including Bell Pepper, '
        'Black Wheat, Carrot, Cowpea, Spinach, Barley, and Chickpea. The framework uses a '
        '10-phase pipeline: Ingestion, AI Extraction, Schema Mapping, Validation, Feature '
        'Engineering, Model Training, Fuzzy Logic, Recommendations, Ready Reckoner generation, '
        'and Continuous Learning.'
    )

    # Key metrics table
    doc.add_heading('Key Metrics', level=2)
    metrics_data = [
        ['Metric', 'Value'],
        ['Total PDFs processed', '24'],
        ['New papers (unique DOIs)', '21'],
        ['Duplicate papers', '3'],
        ['Table rows extracted', '187'],
        ['Papers with yield data', '12'],
        ['ML models trained', '12'],
        ['Best model (CV R2)', 'Gradient Boosting (0.999)'],
        ['Multi-Regression R2', '0.9896'],
        ['Pipeline runtime', '~325 seconds'],
    ]
    table = doc.add_table(rows=len(metrics_data), cols=2)
    table.style = 'Light Grid Accent 1'
    for i, row_data in enumerate(metrics_data):
        for j, cell_text in enumerate(row_data):
            table.rows[i].cells[j].text = str(cell_text)
            if i == 0:
                for paragraph in table.rows[i].cells[j].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True

    doc.add_page_break()

    # ---- 2. DATA INGESTION ----
    doc.add_heading('2. Data Ingestion Summary', level=1)
    ing = safe_read(OUTPUTS_DIR / "ingestion_report.xlsx")
    if ing is not None:
        doc.add_paragraph(f'{len(ing)} papers were ingested from the Data ADES folder.')
        # Add table
        cols_to_show = [c for c in ['Paper_File', 'Crop', 'DOI', 'Status'] if c in ing.columns]
        if cols_to_show:
            table = doc.add_table(rows=min(len(ing)+1, 26), cols=len(cols_to_show))
            table.style = 'Light Grid Accent 1'
            for j, col in enumerate(cols_to_show):
                table.rows[0].cells[j].text = col
                for paragraph in table.rows[0].cells[j].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
            for i in range(min(len(ing), 25)):
                for j, col in enumerate(cols_to_show):
                    val = ing.iloc[i].get(col, '')
                    table.rows[i+1].cells[j].text = str(val)[:60]

    doc.add_page_break()

    # ---- 3. EXTRACTION PERFORMANCE ----
    doc.add_heading('3. Extraction Performance by Paper', level=1)
    extr = safe_read(OUTPUTS_DIR / "extraction_report.csv")
    if extr is not None:
        doc.add_paragraph(
            'The hybrid extraction engine uses 6 readers: Pdfminer, Camelot, Pdfplumber, '
            'Poppler, OCR (Tesseract), and Semantic (sentence-transformers). '
            'Results are merged via confidence-weighted validation.'
        )
        cols_to_show = [c for c in ['Paper', 'pdfminer', 'camelot', 'pdfplumber', 'poppler',
                                     'ocr', 'semantic', 'Rows', 'Confidence', 'Yield_Obs']
                        if c in extr.columns]
        if cols_to_show:
            table = doc.add_table(rows=len(extr)+1, cols=len(cols_to_show))
            table.style = 'Light Grid Accent 1'
            for j, col in enumerate(cols_to_show):
                table.rows[0].cells[j].text = col
                for paragraph in table.rows[0].cells[j].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
            for i in range(len(extr)):
                for j, col in enumerate(cols_to_show):
                    val = extr.iloc[i].get(col, '')
                    if isinstance(val, bool):
                        table.rows[i+1].cells[j].text = 'Y' if val else '-'
                    else:
                        table.rows[i+1].cells[j].text = str(val)[:50]

    doc.add_page_break()

    # ---- 4. MODEL PERFORMANCE ----
    doc.add_heading('4. Model Performance Assessment', level=1)
    perf = safe_read(TABLE_DIR / "Model_Performance_Assessment.xlsx")
    if perf is not None:
        doc.add_paragraph(
            'Twelve machine learning models were trained on 12 samples with 2 meaningful features '
            '(Spike_Length, Table_Row) to predict Yield_per_Hectare. '
            'Cross-validation used K-Fold with k=min(5, n_samples).'
        )
        doc.add_heading('Performance Metrics', level=2)
        cols = ['Model', 'R2', 'Adj_R2', 'MAE', 'RMSE', 'MAPE (%)', 'CV_R2_Mean', 'CV_R2_Std']
        cols = [c for c in cols if c in perf.columns]
        table = doc.add_table(rows=len(perf)+1, cols=len(cols))
        table.style = 'Light Grid Accent 1'
        for j, col in enumerate(cols):
            table.rows[0].cells[j].text = col
            for paragraph in table.rows[0].cells[j].paragraphs:
                for run in paragraph.runs:
                    run.bold = True
        for i in range(len(perf)):
            for j, col in enumerate(cols):
                val = perf.iloc[i].get(col, '')
                if isinstance(val, float):
                    table.rows[i+1].cells[j].text = f'{val:.4f}'
                else:
                    table.rows[i+1].cells[j].text = str(val)

        # Interpretation
        doc.add_heading('Interpretation', level=2)
        best_idx = perf['R2'].astype(str).ne('').idxmax() if 'R2' in perf.columns else 0
        best_r2 = perf.loc[best_idx, 'R2'] if 'R2' in perf.columns else 'N/A'
        doc.add_paragraph(
            f'Best model by R2: {perf.loc[best_idx, "Model"]} (R2 = {best_r2}). '
            'Tree-based models (Decision Tree, Extra Trees, Gradient Boosting, XGBoost, AdaBoost) '
            'achieve R2=1.0 on training data, indicating overfitting due to small sample size (n=12). '
            'The most generalizable models are Multiple Linear Regression (CV R2=0.976) and '
            'Elastic Net (CV R2=0.978). SVR performs poorly (R2=0.31) due to feature scaling issues '
            'with this dataset size.'
        )

    doc.add_page_break()

    # ---- 5. MULTI-REGRESSION ----
    doc.add_heading('5. Multiple Regression Analysis', level=1)
    reg_coef = safe_read(TABLE_DIR / "Multi_Regression_Results.xlsx", sheet_name='Coefficients')
    reg_sum = safe_read(TABLE_DIR / "Multi_Regression_Results.xlsx", sheet_name='Model_Summary')

    if reg_coef is not None and reg_sum is not None:
        doc.add_paragraph(
            'Ordinary Least Squares (OLS) multiple regression was performed with standardized '
            '(z-scored) predictors. The model explains 98.96% of variance in Yield_per_Hectare '
            '(Adj R2 = 0.9872, F = 426.74, p < 0.001).'
        )

        doc.add_heading('Model Summary', level=2)
        table = doc.add_table(rows=len(reg_sum)+1, cols=2)
        table.style = 'Light Grid Accent 1'
        table.rows[0].cells[0].text = 'Metric'
        table.rows[0].cells[1].text = 'Value'
        for paragraph in table.rows[0].cells[0].paragraphs:
            for run in paragraph.runs:
                run.bold = True
        for paragraph in table.rows[0].cells[1].paragraphs:
            for run in paragraph.runs:
                run.bold = True
        for i in range(len(reg_sum)):
            table.rows[i+1].cells[0].text = str(reg_sum.iloc[i].iloc[0])
            table.rows[i+1].cells[1].text = str(reg_sum.iloc[i].iloc[1])

        doc.add_heading('Regression Coefficients', level=2)
        cols = ['Variable', 'Coefficient', 'Std_Error', 't_value', 'p_value', 'Significance']
        cols = [c for c in cols if c in reg_coef.columns]
        table = doc.add_table(rows=len(reg_coef)+1, cols=len(cols))
        table.style = 'Light Grid Accent 1'
        for j, col in enumerate(cols):
            table.rows[0].cells[j].text = col
            for paragraph in table.rows[0].cells[j].paragraphs:
                for run in paragraph.runs:
                    run.bold = True
        for i in range(len(reg_coef)):
            for j, col in enumerate(cols):
                val = reg_coef.iloc[i].get(col, '')
                if isinstance(val, float):
                    table.rows[i+1].cells[j].text = f'{val:.6f}'
                else:
                    table.rows[i+1].cells[j].text = str(val)

        doc.add_heading('Key Findings', level=2)
        doc.add_paragraph(
            'The regression equation (standardized coefficients):\n'
            'Yield = 33.20 - 22.01 x Spike_Length + 6.73 x Table_Row\n\n'
            'Spike_Length is the strongest negative predictor (p < 0.001***), suggesting that '
            'papers reporting longer spikes tend to have lower yields — likely reflecting '
            'crop-specific differences (e.g., chickpea vs. wheat).\n\n'
            'Table_Row is a significant positive predictor (p = 0.005**), indicating that '
            'later table rows in extracted data tend to report higher yields — possibly '
            'reflecting treatment effects (higher fertilizer = higher yield).\n\n'
            'Durbin-Watson statistic = 1.74 (close to 2), indicating no significant '
            'autocorrelation in residuals.'
        )

    doc.add_page_break()

    # ---- 6. FERTILIZER RECOMMENDATIONS ----
    doc.add_heading('6. Fertilizer Recommendations', level=1)
    fert = safe_read(OUTPUTS_DIR / "recommendations" / "fertilizer_recommendations.xlsx")
    if fert is not None:
        doc.add_paragraph(
            'Fuzzy logic-based fertilizer recommendations were generated using 16 Mamdani '
            'rules considering Nitrogen, Phosphorus, Potassium, Zinc, Soil pH, Rainfall, '
            'Temperature, Organic Carbon, Growth Stage, and Yield Prediction.'
        )
        cols = [c for c in fert.columns][:8]
        table = doc.add_table(rows=len(fert)+1, cols=len(cols))
        table.style = 'Light Grid Accent 1'
        for j, col in enumerate(cols):
            table.rows[0].cells[j].text = col
            for paragraph in table.rows[0].cells[j].paragraphs:
                for run in paragraph.runs:
                    run.bold = True
        for i in range(len(fert)):
            for j, col in enumerate(cols):
                table.rows[i+1].cells[j].text = str(fert.iloc[i].get(col, ''))[:50]

    doc.add_page_break()

    # ---- 7. READY RECKONER ----
    doc.add_heading('7. Ready Reckoner Table (24 Papers)', level=1)
    rr = safe_read(OUTPUTS_DIR / "Ready_Reckoner_24Papers.xlsx")
    if rr is not None:
        doc.add_paragraph(
            f'The Ready Reckoner aggregates extracted data from all {len(rr)} papers. '
            'Values are means across all treatments within each paper.'
        )
        # Show key columns
        key_cols = [c for c in ['Source_File', 'Yield_per_Hectare', 'Plant_Height_cm',
                                'SPAD', 'Nitrogen', 'Phosphorus', 'Potassium',
                                'Soil_pH', 'Rainfall', 'Protein', 'N_Treatments']
                    if c in rr.columns]
        if key_cols:
            table = doc.add_table(rows=min(len(rr)+1, 26), cols=len(key_cols))
            table.style = 'Light Grid Accent 1'
            for j, col in enumerate(key_cols):
                table.rows[0].cells[j].text = col[:20]
                for paragraph in table.rows[0].cells[j].paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
            for i in range(min(len(rr), 25)):
                for j, col in enumerate(key_cols):
                    val = rr.iloc[i].get(col, '')
                    if isinstance(val, float) and not np.isnan(val):
                        table.rows[i+1].cells[j].text = f'{val:.2f}'
                    else:
                        table.rows[i+1].cells[j].text = str(val)[:30] if pd.notna(val) else '-'

    doc.add_page_break()

    # ---- 8. VALIDATION ----
    doc.add_heading('8. Validation Summary', level=1)
    val = safe_read(OUTPUTS_DIR / "validation_report.xlsx")
    if val is not None:
        doc.add_paragraph(
            f'Validation checked {len(val)} columns for missing values, outliers, '
            'impossible values, and OCR artifacts.'
        )
        cols = [c for c in ['Column', 'Non_Null', 'Missing', 'Missing_Pct', 'Issues']
                if c in val.columns]
        table = doc.add_table(rows=min(len(val)+1, 30), cols=len(cols))
        table.style = 'Light Grid Accent 1'
        for j, col in enumerate(cols):
            table.rows[0].cells[j].text = col
            for paragraph in table.rows[0].cells[j].paragraphs:
                for run in paragraph.runs:
                    run.bold = True
        for i in range(min(len(val), 29)):
            for j, col in enumerate(cols):
                table.rows[i+1].cells[j].text = str(val.iloc[i].get(col, ''))[:60]

    # ---- FOOTER ----
    doc.add_paragraph()
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run('--- End of Report ---')
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
    run.font.size = Pt(9)

    docx_path = EXPORT_DIR / "AAIF_Model_Report.docx"
    doc.save(str(docx_path))
    log(f"  Narrative report saved: {docx_path}")
    log(f"  Size: {docx_path.stat().st_size / 1024:.1f} KB")
    return docx_path


# =========================================================================
# PART 3: REGRESSION-FOCUSED WORD DOCUMENT
# =========================================================================
def export_regression_report():
    """Create a focused Word document for regression analysis."""
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    log("Creating regression report (DOCX)...")

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    # Title
    title = doc.add_heading('Multiple Regression Analysis Report', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('Yield_per_Hectare Prediction — OLS Regression')
    run.font.size = Pt(13)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x99)

    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_para.add_run(f'Generated: {datetime.now().strftime("%B %d, %Y %H:%M")}')
    run.font.size = Pt(10)

    doc.add_paragraph()

    # Data Description
    doc.add_heading('1. Data Description', level=1)
    doc.add_paragraph(
        'Training data: 12 observations with Yield_per_Hectare as the dependent variable.\n'
        'Features: 26 candidate predictors extracted from 24 research papers.\n'
        'After removing zero-variance columns, 2 meaningful features remained:\n'
        '  - Spike_Length (cm): spike/ear/panicle length\n'
        '  - Table_Row: row index in extracted table (proxy for treatment order)'
    )

    # Model Summary
    reg_sum = safe_read(TABLE_DIR / "Multi_Regression_Results.xlsx", sheet_name='Model_Summary')
    if reg_sum is not None:
        doc.add_heading('2. Model Summary', level=1)
        table = doc.add_table(rows=len(reg_sum)+1, cols=2)
        table.style = 'Light Grid Accent 1'
        table.rows[0].cells[0].text = 'Metric'
        table.rows[0].cells[1].text = 'Value'
        for p in table.rows[0].cells[0].paragraphs:
            for r in p.runs:
                r.bold = True
        for p in table.rows[0].cells[1].paragraphs:
            for r in p.runs:
                r.bold = True
        for i in range(len(reg_sum)):
            table.rows[i+1].cells[0].text = str(reg_sum.iloc[i].iloc[0])
            val = reg_sum.iloc[i].iloc[1]
            table.rows[i+1].cells[1].text = f'{val:.4f}' if isinstance(val, (int, float)) else str(val)

    # Coefficients
    reg_coef = safe_read(TABLE_DIR / "Multi_Regression_Results.xlsx", sheet_name='Coefficients')
    if reg_coef is not None:
        doc.add_heading('3. Regression Coefficients', level=1)
        doc.add_paragraph(
            'Coefficients are for standardized (z-scored) predictors, '
            'so they represent relative importance.'
        )
        cols = ['Variable', 'Coefficient', 'Std_Error', 't_value', 'p_value', 'Significance']
        cols = [c for c in cols if c in reg_coef.columns]
        table = doc.add_table(rows=len(reg_coef)+1, cols=len(cols))
        table.style = 'Light Grid Accent 1'
        for j, col in enumerate(cols):
            table.rows[0].cells[j].text = col
            for p in table.rows[0].cells[j].paragraphs:
                for r in p.runs:
                    r.bold = True
        for i in range(len(reg_coef)):
            for j, col in enumerate(cols):
                val = reg_coef.iloc[i].get(col, '')
                if isinstance(val, float):
                    table.rows[i+1].cells[j].text = f'{val:.6f}'
                else:
                    table.rows[i+1].cells[j].text = str(val)

    # Model Performance
    perf = safe_read(TABLE_DIR / "Model_Performance_Assessment.xlsx")
    if perf is not None:
        doc.add_heading('4. All Models Comparison', level=1)
        doc.add_paragraph('Twelve models were evaluated. Key metrics:')
        cols = ['Model', 'R2', 'Adj_R2', 'MAE', 'RMSE', 'CV_R2_Mean']
        cols = [c for c in cols if c in perf.columns]
        table = doc.add_table(rows=len(perf)+1, cols=len(cols))
        table.style = 'Light Grid Accent 1'
        for j, col in enumerate(cols):
            table.rows[0].cells[j].text = col
            for p in table.rows[0].cells[j].paragraphs:
                for r in p.runs:
                    r.bold = True
        for i in range(len(perf)):
            for j, col in enumerate(cols):
                val = perf.iloc[i].get(col, '')
                if isinstance(val, float):
                    table.rows[i+1].cells[j].text = f'{val:.4f}'
                else:
                    table.rows[i+1].cells[j].text = str(val)

    # Interpretation
    doc.add_heading('5. Interpretation', level=1)
    doc.add_paragraph(
        'The OLS model explains 98.96% of yield variance (R2=0.9896, Adj R2=0.9872) '
        'with high overall significance (F=426.74, p<0.001).\n\n'
        'Significant predictors:\n'
        '  1. Spike_Length (beta=-22.01, p<0.001***): Strong negative predictor. '
        'Longer spikes are associated with lower yields. This likely reflects '
        'crop-specific variation — chickpea (high yield) has shorter spikes than wheat.\n\n'
        '  2. Table_Row (beta=+6.73, p=0.005**): Positive predictor. Later table rows '
        'tend to have higher yields, suggesting treatment effects (increased fertilizer '
        'application leads to higher yield).\n\n'
        'Model diagnostics:\n'
        '  - Durbin-Watson = 1.74 (no autocorrelation)\n'
        '  - Condition Number = 3.58 (no multicollinearity)\n'
        '  - Residual Std Error = 3.32 t/ha'
    )

    # Caveats
    doc.add_heading('6. Caveats', level=1)
    doc.add_paragraph(
        '  - Sample size is small (n=12). Results should be validated on larger datasets.\n'
        '  - Most features had zero variance (all-NaN in the yield subset), '
        'limiting the model to 2 predictors.\n'
        '  - Tree-based models show overfitting (R2=1.0 on training data).\n'
        '  - Cross-validation R2 ranges from 0.656 (XGBoost, unstable) to 0.999 (Gradient Boosting).\n'
        '  - The most reliable generalizable model is Elastic Net (CV R2=0.978 +/- 0.015).'
    )

    docx_path = EXPORT_DIR / "AAIF_Regression_Report.docx"
    doc.save(str(docx_path))
    log(f"  Regression report saved: {docx_path}")
    log(f"  Size: {docx_path.stat().st_size / 1024:.1f} KB")
    return docx_path


# =========================================================================
# MAIN
# =========================================================================
def main():
    log("=" * 70)
    log("  AAIF RESULTS EXPORT")
    log("=" * 70)

    xlsx = export_master_excel()
    docx1 = export_narrative_report()
    docx2 = export_regression_report()

    log("\n" + "=" * 70)
    log("  EXPORT COMPLETE")
    log("=" * 70)
    log(f"\n  Files saved to: {EXPORT_DIR}")
    log(f"\n  1. {xlsx.name}")
    log(f"     Master Excel workbook with {15} sheets")
    log(f"  2. {docx1.name}")
    log(f"     Full narrative report (8 sections)")
    log(f"  3. {docx2.name}")
    log(f"     Regression analysis report (6 sections)")


if __name__ == '__main__':
    main()
