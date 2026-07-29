import json
from datetime import datetime
from pathlib import Path

from agri_ai_agent.continuous_learning.change_detector import ChangeSet
from agri_ai_agent.continuous_learning.version_history import VersionHistory


def generate_sync_report(changes: ChangeSet, affected_stages: list[str],
                         output_dir: Path) -> Path:
    report_dir = output_dir / "continuous_learning"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report_type": "synchronization",
        "generated_at": datetime.now().isoformat(),
        "changes": changes.to_dict(),
        "affected_stages": affected_stages,
        "stages_count": len(affected_stages),
        "repos_with_changes": changes.changed_repos,
        "data_changed": changes.has_data_changes,
        "model_changed": changes.has_model_changes,
    }

    path = report_dir / f"sync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    md_path = report_dir / f"sync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    md_lines = [
        "# Synchronization Report",
        f"Generated: {report['generated_at']}",
        "",
        "## Changes Detected",
        f"- Repos changed: {len(changes.changed_repos)}",
        f"- New datasets: {len(changes.new_datasets)}",
        f"- Changed observations: {len(changes.changed_observations)}",
        f"- Changed features: {len(changes.changed_features)}",
        f"- New models: {len(changes.new_model_versions)}",
        "",
        "## Affected Stages",
    ]
    for stage in affected_stages:
        md_lines.append(f"- {stage}")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return path


def generate_provenance_report(history: VersionHistory, output_dir: Path) -> Path:
    report_dir = output_dir / "continuous_learning"
    report_dir.mkdir(parents=True, exist_ok=True)

    syncs = history.list_syncs(limit=5)
    report = {
        "report_type": "provenance",
        "generated_at": datetime.now().isoformat(),
        "recent_syncs": syncs,
        "syncs_count": len(syncs),
    }

    path = report_dir / f"provenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def generate_retraining_report(stats: dict, output_dir: Path) -> Path:
    report_dir = output_dir / "continuous_learning"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "report_type": "retraining",
        "generated_at": datetime.now().isoformat(),
        "cycle_stats": stats,
        "drift_detected": stats.get("drift_detected", False),
        "model_retrained": stats.get("model_retrained", False),
        "reckoner_updated": stats.get("reckoner_updated", False),
    }

    path = report_dir / f"retraining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    md_path = report_dir / f"retraining_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    md_lines = [
        "# Retraining Report",
        f"Generated: {report['generated_at']}",
        "",
        "## Cycle Summary",
        f"- Data loaded: {stats.get('data_loaded', False)}",
        f"- Features engineered: {stats.get('features_engineered', False)}",
        f"- Drift detected: {stats.get('drift_detected', False)}",
        f"- Model retrained: {stats.get('model_retrained', False)}",
        f"- Reckoner updated: {stats.get('reckoner_updated', False)}",
        f"- Duration: {stats.get('duration_sec', 0):.1f}s",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return path
