"""Phase 20 orchestrator — dry run then production, checkpoint-resumable.

Run modes:
    python p20_run_all.py            dry run (config dry_run=True)
    python p20_run_all.py --prod     production run

Pipeline order (STEPs 0-26):
    step0_path_audit, step1_input_manifest,
    identity(2), spatial(3), temporal(4), ontology(5),
    external_linkage(6-8), predictors(9-10), completeness(11),
    information_gain(12,18), independence(14), leakage(15),
    dataset(13,16,17,19), rag_sync(20), reports(21),
    step23_quality_gates, step24_metrics, step25_version_manifest,
    step26_final_decision.

Protected inputs are checksummed before and after; any drift is reported.
"""

from __future__ import annotations

import sys
import time

try:
    from . import p20_common as C
except ImportError:
    import p20_common as C


def _import(mod):
    if str(C.TMP) not in sys.path:
        sys.path.insert(0, str(C.TMP))
    m = __import__(mod)
    return m


STAGES = [
    ("step0_path_audit", "p20_validation", "step0_path_audit"),
    ("step1_input_manifest", "p20_validation", "step1_input_manifest"),
    ("identity", "p20_identity", "run"),
    ("spatial", "p20_spatial", "run"),
    ("temporal", "p20_temporal", "run"),
    ("ontology", "p20_ontology", "run"),
    ("external_linkage", "p20_external_linkage", "run"),
    ("predictors", "p20_predictors", "run"),
    ("completeness", "p20_completeness", "run"),
    ("information_gain", "p20_information_gain", "run"),
    ("independence", "p20_independence", "run"),
    ("leakage", "p20_leakage", "run"),
    ("dataset", "p20_dataset", "run"),
    ("rag_sync", "p20_rag", "run"),
    ("step23_quality_gates", "p20_validation", "step23_quality_gates"),
    ("step24_metrics", "p20_validation", "step24_metrics"),
    ("step25_version_manifest", "p20_validation", "step25_version_manifest"),
    ("step26_final_decision", "p20_validation", "step26_final_decision"),
    ("reports", "p20_reports", "run"),
]


def _before_checksums(cfg):
    sums = C.protected_checksums(only_existing=True)
    C.write_json(
        {
            "mode": "dry_run" if cfg.get("dry_run") else "production",
            "recorded_at": C.now_full_iso(),
            "checksums": sums,
        },
        C.OUT / "protected_input_checksums_before.json",
    )
    return sums


def _after_checksums(cfg):
    sums = C.protected_checksums(only_existing=True)
    C.write_json(
        {
            "mode": "dry_run" if cfg.get("dry_run") else "production",
            "recorded_at": C.now_full_iso(),
            "checksums": sums,
        },
        C.OUT / "protected_input_checksums_after.json",
    )
    return sums


def _compare_checksums():
    bef = C.read_json(C.OUT / "protected_input_checksums_before.json", {}).get("checksums", {})
    aft = C.read_json(C.OUT / "protected_input_checksums_after.json", {}).get("checksums", {})
    drift = {k: (bef.get(k), aft.get(k)) for k in bef if bef.get(k) != aft.get(k)}
    result = {"drift_detected": bool(drift), "drifted_files": drift}
    C.write_json(result, C.OUT / "protected_checksum_verification.json")
    return result


def run_pipeline(dry_run: bool):
    cfg = C.load_config()
    cfg["dry_run"] = dry_run

    C.log_msg(f"Phase 20 run starting: mode={'DRY' if dry_run else 'PROD'}")
    _before_checksums(cfg)

    results = {}
    for stage, module, func in STAGES:
        if C.is_done(stage):
            C.log_msg(f"skip {stage} (already done)")
            continue
        if stage == "step23_quality_gates":
            # record + compare protected checksums BEFORE running the gates so
            # the gate has real drift evidence from the just-completed stages
            _after_checksums(cfg)
            _compare_checksums()
        mod = _import(module)
        fn = getattr(mod, func)
        C.log_msg(f"running {stage} ({module}.{func})")
        t0 = time.time()
        try:
            r = fn(force=False)
            results[stage] = {"ok": True, "elapsed_s": round(time.time() - t0, 2), "result": r}
        except Exception as exc:
            C.log_msg(f"FAILED {stage}: {exc}")
            results[stage] = {
                "ok": False,
                "elapsed_s": round(time.time() - t0, 2),
                "error": str(exc),
            }
            raise

    _after_checksums(cfg)
    drift = _compare_checksums()

    report = {
        "mode": "DRY" if dry_run else "PROD",
        "generated_at": C.now_full_iso(),
        "stages": results,
        "protected_checksum_drift": drift,
        "final_decision": C.read_json(C.OUT / "final_decision.json", {}),
    }
    C.write_json(report, C.OUT / "run_report.json")
    C.write_json(report, C.REPORTS / "run_report.json")
    C.log_msg(f"Phase 20 finished: {len(results)} stages, drift={drift['drift_detected']}")
    return report


def main():
    args = sys.argv[1:]
    prod = "--prod" in args or "-p" in args
    if "--dry" in args:
        prod = False

    print(f"[phase20] PROJECT_ROOT = {C.PROJECT_ROOT}")
    print(f"[phase20] mode = {'PRODUCTION' if prod else 'DRY RUN'}")

    if prod:
        print("[phase20] Running production pipeline...")
        rep = run_pipeline(dry_run=False)
    else:
        print("[phase20] DRY RUN — running the full pipeline in dry-run mode")
        print("[phase20] (all outputs land in outputs/phase20, checkpointed)")
        rep = run_pipeline(dry_run=True)

    print(f"[phase20] done. stages executed: {len(rep['stages'])}")
    print(f"[phase20] checksum drift: {rep['protected_checksum_drift']}")
    print(f"[phase20] decision: {rep.get('final_decision', {}).get('decision')}")
    return 0 if not rep.get("protected_checksum_drift", {}).get("drift_detected") else 2


if __name__ == "__main__":
    sys.exit(main())
