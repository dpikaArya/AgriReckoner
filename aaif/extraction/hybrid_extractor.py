import os, sys, time, json
import pandas as pd

try:
    from agri_ai_agent.config.schema import UAMS_COLUMNS
except ImportError:
    from . import UAMS_COLUMNS

from .pdfminer_reader import PdfminerReader
from .camelot_reader import CamelotReader
from .pdfplumber_reader import PdfplumberReader
from .poppler_reader import PopplerReader
from .ocr_reader import OcrReader
from .semantic_extractor import SemanticExtractor

try:
    from agri_ai_agent.agents.evidence_fusion_agent import EvidenceFusionAgent as ValidationAgent
except ImportError:
    from .validation_agent import ValidationAgent

class HybridExtractor:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        self.readers = [
            PdfminerReader(),
            CamelotReader(),
            PdfplumberReader(),
            PopplerReader(),
            OcrReader(),
            SemanticExtractor(),
        ]
        self.validator = ValidationAgent()
        self.results = {}

    def run(self, input_dir):
        if not os.path.isdir(input_dir):
            print(f"ERROR: input directory not found: {input_dir}")
            return

        pdf_files = sorted([f for f in os.listdir(input_dir)
                           if f.lower().endswith('.pdf')])
        print(f"\n{'='*70}")
        print(f"  HYBRID EXTRACTION ENGINE")
        print(f"  Input: {input_dir}")
        print(f"  PDFs found: {len(pdf_files)}")
        print(f"{'='*70}\n")

        all_paper_results = []
        overall_start = time.time()

        for pdf_name in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_name)
            paper_result = self._process_paper(pdf_path, pdf_name)
            all_paper_results.append(paper_result)

        dt = time.time() - overall_start
        self.results = {
            'papers': all_paper_results,
            'total_time_s': round(dt, 1),
        }

        self._print_summary(all_paper_results, pdf_files, dt)
        self._save_outputs(all_paper_results)
        self._save_report(all_paper_results, dt)

    def _process_paper(self, pdf_path, pdf_name):
        print(f"\n{'─'*60}")
        print(f"  [{pdf_name}]")
        print(f"{'─'*60}")

        stage_results = []

        is_scanned = False
        poppler_reader = next((r for r in self.readers if r.name == 'poppler'), None)
        if poppler_reader and poppler_reader.available:
            try:
                is_scanned = poppler_reader._detect_scanned_pdf(pdf_path)
                if is_scanned:
                    print(f"    {'SCANNED PDF DETECTED':20s} -- routing to OCR first")
            except Exception:
                pass

        ordered_readers = self.readers[:]
        if is_scanned:
            ocr_idx = next((i for i, r in enumerate(self.readers) if r.name == 'ocr'), None)
            poppler_idx = next((i for i, r in enumerate(self.readers) if r.name == 'poppler'), None)
            if ocr_idx is not None:
                ordered_readers.remove(self.readers[ocr_idx])
                ordered_readers.insert(0, self.readers[ocr_idx])
            if poppler_idx is not None:
                ordered_readers.remove(self.readers[poppler_idx])
                ordered_readers.insert(1, self.readers[poppler_idx])

        for reader in ordered_readers:
            t0 = time.time()
            try:
                result = reader.extract(pdf_path)
                dt = time.time() - t0
                if result.get('error'):
                    print(f"    {reader.name:20s} ✗ {dt:5.1f}s  {result['error'][:50]}")
                elif result.get('success'):
                    print(f"    {reader.name:20s} ✓ {dt:5.1f}s  {len(result.get('rows', [])):3d} rows, conf={result.get('confidence', 0):.2f}")
                else:
                    print(f"    {reader.name:20s} – {dt:5.1f}s  no data")
                stage_results.append(result)
            except Exception as e:
                print(f"    {reader.name:20s} ✗ 0.0s  {str(e)[:50]}")
                stage_results.append({
                    'reader': reader.name, 'success': False,
                    'rows': [], 'confidence': 0.0, 'error': str(e)
                })

        for sr in stage_results:
            if sr.get('rows'):
                for r in sr['rows']:
                    if 'Source_File' not in r or not r['Source_File']:
                        r['Source_File'] = pdf_name

        if hasattr(self.validator, 'process'):
            result_df = self.validator.process(
                pd.DataFrame(), stage_results=stage_results, source_file=pdf_path
            )
            merged_rows = result_df.to_dict('records') if not result_df.empty else []
            merged = {
                'success': True,
                'rows': merged_rows,
                'confidence': float(result_df['_fused_confidence'].mean()) if '_fused_confidence' in result_df.columns and not result_df.empty else 0.0,
            }
        else:
            merged = self.validator.merge(stage_results, source_file=pdf_path)
            merged['rows'] = self.validator.normalize_units(merged.get('rows', []))
            merged['rows'] = self.validator.validate_duplicates(merged.get('rows', []))
            merged['rows'] = self.validator.validate_crop_consistency(merged.get('rows', []))
            merged['rows'] = self.validator.validate_treatment_ids(merged.get('rows', []))

        n_yield = sum(1 for r in merged.get('rows', [])
                     if any(r.get(yc) is not None for yc in
                            ['Yield_per_Hectare', 'Yield_per_Plot', 'Yield_per_Acre',
                             'Biomass_Yield', 'Harvest_Index']))

        print(f"    {'VALIDATED':20s} ✓ --  {len(merged.get('rows', [])):3d} rows, {n_yield} with yield")

        return {
            'pdf': pdf_name,
            'stage_results': stage_results,
            'merged': merged,
        }

    def _print_summary(self, paper_results, pdf_files, dt):
        print(f"\n{'='*70}")
        print(f"  EXTRACTION SUMMARY")
        print(f"{'='*70}")
        print(f"\n  PDFs processed: {len(paper_results)}/{len(pdf_files)}")
        print(f"  Total time: {dt:.1f}s")

        total_rows = 0
        total_yield = 0
        total_papers_with_data = 0
        stage_counts = {}

        for pr in paper_results:
            n = len(pr['merged'].get('rows', []))
            total_rows += n
            ny = sum(1 for r in pr['merged'].get('rows', [])
                    if any(r.get(yc) is not None for yc in
                           ['Yield_per_Hectare', 'Yield_per_Plot', 'Yield_per_Acre',
                            'Biomass_Yield', 'Harvest_Index']))
            total_yield += ny
            if n > 0:
                total_papers_with_data += 1

            for sr in pr['stage_results']:
                rname = sr.get('reader', '?')
                stage_counts.setdefault(rname, {'success': 0, 'total': 0, 'rows': 0})
                stage_counts[rname]['total'] += 1
                if sr.get('success'):
                    stage_counts[rname]['success'] += 1
                stage_counts[rname]['rows'] += len(sr.get('rows', []))

        print(f"  Papers with extracted data: {total_papers_with_data}")
        print(f"  Total merged rows: {total_rows}")
        print(f"  Total yield observations: {total_yield}")

        print(f"\n  Stage performance:")
        for rname, cnts in sorted(stage_counts.items()):
            pct = cnts['success'] / cnts['total'] * 100 if cnts['total'] else 0
            print(f"    {rname:20s}: {cnts['success']}/{cnts['total']} success ({pct:.0f}%), {cnts['rows']} rows")

        # Compute completeness
        if total_rows > 0:
            print(f"\n  Variable completeness:")
            all_rows = []
            for pr in paper_results:
                all_rows.extend(pr['merged'].get('rows', []))
            if all_rows:
                df = pd.DataFrame(all_rows)
                for col in ['Yield_per_Hectare', 'Yield_per_Plot', 'Biomass_Yield',
                            'Plant_Height_cm', 'SPAD', 'Tillers', '100_Seed_Weight',
                            'Nitrogen', 'Phosphorus', 'Potassium', 'Organic_Carbon']:
                    if col in df.columns:
                        n = df[col].notna().sum()
                        missing = (1 - n / len(df)) * 100
                        print(f"    {col:25s}: {n:4d} / {len(df):4d} ({100-missing:.0f}%)")

        print(f"\n  Output files in: {self.output_dir}")

    def _save_outputs(self, paper_results):
        all_rows = []
        for pr in paper_results:
            for r in pr['merged'].get('rows', []):
                r['Source_File'] = pr['pdf']
                all_rows.append(r)

        if not all_rows:
            print("  WARNING: no rows to save")
            return

        df = self.validator.fill_uas_columns(all_rows)
        df = self.validator.to_dataframe(df)

        uas_path = os.path.join(self.output_dir, 'Universal_Agricultural_Schema.xlsx')
        self.validator.map_to_uas(df, uas_path)
        print(f"  UAS schema saved: {uas_path}")

        treatment_path = os.path.join(self.output_dir, 'hybrid_extraction_results.xlsx')
        df.to_excel(treatment_path, index=False)
        print(f"  Extraction results: {treatment_path}")

    def _save_report(self, paper_results, dt):
        import csv
        report_path = os.path.join(self.output_dir, 'extraction_report.csv')
        fields = ['Paper', 'pdfminer', 'camelot', 'pdfplumber', 'poppler', 'ocr', 'semantic',
                  'Rows', 'Confidence', 'Yield_Obs', 'Missing_Vars']

        rows_out = []
        for pr in paper_results:
            meta = {sr.get('reader', '?'): sr.get('success', False)
                    for sr in pr['stage_results']}
            confs = [sr.get('confidence', 0) for sr in pr['stage_results'] if sr.get('success')]
            avg_conf = sum(confs) / len(confs) if confs else 0

            merged_rows = pr['merged'].get('rows', [])
            n_yield = sum(1 for r in merged_rows
                         if any(r.get(yc) is not None for yc in
                                ['Yield_per_Hectare', 'Yield_per_Plot', 'Yield_per_Acre',
                                 'Biomass_Yield', 'Harvest_Index']))

            rows_out.append({
                'Paper': pr['pdf'],
                'pdfminer': meta.get('pdfminer', False),
                'camelot': meta.get('camelot', False),
                'pdfplumber': meta.get('pdfplumber', False),
                'poppler': meta.get('poppler', False),
                'ocr': meta.get('ocr', False),
                'semantic': meta.get('semantic', False),
                'Rows': len(merged_rows),
                'Confidence': round(avg_conf, 2),
                'Yield_Obs': n_yield,
                'Missing_Vars': '',
            })

        with open(report_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"  Extraction report: {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hybrid Research Paper Extraction Engine')
    parser.add_argument('input_dir', help='Directory containing PDF files')
    parser.add_argument('--output', '-o', default=None, help='Output directory')
    args = parser.parse_args()

    engine = HybridExtractor(output_dir=args.output)
    engine.run(args.input_dir)


if __name__ == '__main__':
    main()
