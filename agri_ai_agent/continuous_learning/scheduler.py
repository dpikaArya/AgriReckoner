"""Scheduler policy for the continuous-learning loop.

Policies live in ``config/scheduler.yaml`` (externalized, loaded via
``LiteratureConfig.from_env().scheduler``).  This module only decides *whether*
a run is due — it never executes the pipeline.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

_UNSET = object()


def load_scheduler_config() -> dict:
    """Load the externalized scheduler policy (empty dict when unavailable)."""
    try:
        from agri_ai_agent.literature.config import LiteratureConfig

        return dict(LiteratureConfig.from_env().scheduler or {})
    except Exception:  # noqa: BLE001 - scheduler is advisory
        return {}


def _as_float(value, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def scheduled_run_due(
    last_run_at: datetime | None,
    scheduler_cfg: dict | None = None,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Return ``(due, reason)`` for a scheduled continuous-learning run.

    ``last_run_at`` is the wall-clock time of the previous run (UTC), or None
    for a first run.  With no scheduler policy the run is always due.
    """
    cfg = scheduler_cfg if scheduler_cfg is not None else load_scheduler_config()
    if not cfg:
        return True, "no scheduler policy configured; running on demand"

    if not _as_float(cfg.get("enabled"), 1.0):
        return False, "scheduler disabled in config/scheduler.yaml"

    now = now or datetime.now(timezone.utc)

    interval_sec = _as_float(cfg.get("interval_sec"), 86400.0)
    if interval_sec and interval_sec > 0 and last_run_at is not None:
        if last_run_at.tzinfo is None:
            last_run_at = last_run_at.replace(tzinfo=timezone.utc)
        if (now - last_run_at) < timedelta(seconds=interval_sec):
            return False, f"last run {last_run_at.isoformat()} within interval ({interval_sec:.0f}s)"

    run_at = cfg.get("run_at")
    if run_at:
        try:
            hour, minute = (int(p) for p in str(run_at).split(":", 1))
        except (ValueError, TypeError):
            hour = minute = None
        if hour is not None:
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if (now - target) < timedelta(minutes=-5):  # not yet reached today
                return False, f"scheduled run_at {run_at} not reached yet"

    max_runs = _as_float(cfg.get("max_runs_per_day"), 1.0)
    if max_runs <= 0:
        return False, "scheduler max_runs_per_day <= 0"
    return True, "scheduled run is due"


def last_run_from_history(history) -> datetime | None:
    """Best-effort last-run timestamp from a VersionHistory instance."""
    try:
        syncs = history.list_syncs(limit=1)
        if syncs and syncs[0].get("started_at"):
            return datetime.fromisoformat(syncs[0]["started_at"])
    except Exception:  # noqa: BLE001
        pass
    return None
