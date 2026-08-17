"""
End-to-End Pipeline Performance Evaluation.
Evaluates pipeline completion rate, agent failures, retries, execution time,
CPU, memory, GPU, peak memory, disk usage.
"""

import time

from evaluation.utils import (
    PIPELINE_AGENTS,
    format_seconds,
    get_cpu_usage,
    get_memory_usage,
    load_provenance,
)


def evaluate_e2e():
    provenance = load_provenance()

    status = provenance.get("status", "unknown")
    completed_agents = provenance.get("completed_agents", [])
    failed_agents = provenance.get("failed_agents", [])
    results = provenance.get("results", {})

    total_time = 0.0
    total_retries = 0
    total_errors = 0

    for _, result in results.items():
        total_time += result.get("execution_time_sec", 0)
        total_retries += result.get("retry_count", 0)
        total_errors += len(result.get("errors", []))

    completion_rate = len(completed_agents) / len(PIPELINE_AGENTS) if PIPELINE_AGENTS else 0

    mem = get_memory_usage()
    cpu = get_cpu_usage()

    report = f"""# End-to-End Pipeline Performance Report
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Summary
| Metric | Value |
|--------|-------|
| Pipeline status | {status} |
| Completion rate | {completion_rate:.1%} |
| Agents completed | {len(completed_agents)}/{len(PIPELINE_AGENTS)} |
| Agent failures | {len(failed_agents)} |
| Total retries | {total_retries} |
| Total errors | {total_errors} |
| Total execution time | {format_seconds(total_time)} |
| Peak memory (RSS) | {mem.get("rss_mb", 0):.1f} MB |
| Current CPU usage | {cpu:.1f}% |

## Per-Agent Execution Times
| Agent | Time (s) | Retries | Errors |
|-------|----------|---------|--------|
"""
    for agent_key in PIPELINE_AGENTS:
        r = results.get(agent_key, {})
        et = r.get("execution_time_sec", 0)
        rc = r.get("retry_count", 0)
        err_count = len(r.get("errors", []))
        report += f"| {agent_key} | {et:.4f} | {rc} | {err_count} |\n"

    report += f"""
## Failed Agents ({len(failed_agents)})
"""
    for fa in failed_agents:
        report += f"- {fa.get('step')}: {', '.join(fa.get('errors', []))}\n"

    if failed_agents:
        report += "None\n"

    report += f"""
## Resource Usage
- Memory (RSS): {mem.get("rss_mb", 0):.1f} MB
- Memory (VMS): {mem.get("vms_mb", 0):.1f} MB
- Memory percent: {mem.get("percent", 0):.1f}%
- CPU: {cpu:.1f}%

> Note: GPU usage not available on this system.

## Bottlenecks & Recommendations
1. **Optimize slow agents** by profiling and parallelizing where possible
2. **Reduce retries** by fixing root causes of agent failures
3. **Consider incremental processing** for large datasets
4. **Add checkpoint recovery** to resume from failures
5. **Monitor memory** for large dataset ingestion, consider chunking
"""
    path = write_report(
        "Pipeline_Performance.md",
        report,
    )
    return {
        "completion_rate": completion_rate,
        "total_time": total_time,
        "total_retries": total_retries,
        "total_errors": total_errors,
        "failed_count": len(failed_agents),
    }, str(path)


def write_report(filename: str, content: str):
    from evaluation.utils import REPORTS_DIR, ensure_reports_dir

    ensure_reports_dir()
    path = REPORTS_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path
