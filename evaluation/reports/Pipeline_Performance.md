# End-to-End Pipeline Performance Report
Generated: 2026-07-28 15:48:25

## Summary
| Metric | Value |
|--------|-------|
| Pipeline status | completed |
| Completion rate | 100.0% |
| Agents completed | 12/12 |
| Agent failures | 0 |
| Total retries | 0 |
| Total errors | 0 |
| Total execution time | 8.95m |
| Peak memory (RSS) | 410.4 MB |
| Current CPU usage | 0.0% |

## Per-Agent Execution Times
| Agent | Time (s) | Retries | Errors |
|-------|----------|---------|--------|
| ingestion | 511.0000 | 0 | 0 |
| schema_mapping | 0.1000 | 0 | 0 |
| ontology_mapping | 0.0000 | 0 | 0 |
| unit_harmonization | 0.0000 | 0 | 0 |
| quality_assurance | 0.0000 | 0 | 0 |
| feature_engineering | 0.1000 | 0 | 0 |
| leakage_detection | 0.0000 | 0 | 0 |
| encoding | 0.0000 | 0 | 0 |
| statistical_diagnostics | 0.0000 | 0 | 0 |
| model_readiness | 0.0000 | 0 | 0 |
| documentation | 0.5000 | 0 | 0 |
| export | 0.5000 | 0 | 0 |

## Failed Agents (0)

## Resource Usage
- Memory (RSS): 410.4 MB
- Memory (VMS): 610.9 MB
- Memory percent: 2.4%
- CPU: 0.0%

> Note: GPU usage not available on this system.

## Bottlenecks & Recommendations
1. **Optimize slow agents** by profiling and parallelizing where possible
2. **Reduce retries** by fixing root causes of agent failures
3. **Consider incremental processing** for large datasets
4. **Add checkpoint recovery** to resume from failures
5. **Monitor memory** for large dataset ingestion, consider chunking
