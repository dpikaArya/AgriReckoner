# End-to-End Pipeline Performance Report
Generated: 2026-07-04 22:30:10

## Summary
| Metric | Value |
|--------|-------|
| Pipeline status | completed |
| Completion rate | 100.0% |
| Agents completed | 12/12 |
| Agent failures | 0 |
| Total retries | 0 |
| Total errors | 0 |
| Total execution time | 1.99s |
| Peak memory (RSS) | 72.0 MB |
| Current CPU usage | 0.1% |

## Per-Agent Execution Times
| Agent | Time (s) | Retries | Errors |
|-------|----------|---------|--------|
| ingestion | 0.0042 | 0 | 0 |
| schema_mapping | 0.0536 | 0 | 0 |
| ontology_mapping | 0.0047 | 0 | 0 |
| unit_harmonization | 0.0033 | 0 | 0 |
| quality_assurance | 0.1293 | 0 | 0 |
| feature_engineering | 0.0331 | 0 | 0 |
| leakage_detection | 0.0358 | 0 | 0 |
| encoding | 0.0369 | 0 | 0 |
| statistical_diagnostics | 0.6239 | 0 | 0 |
| model_readiness | 0.0068 | 0 | 0 |
| documentation | 0.3966 | 0 | 0 |
| export | 0.6622 | 0 | 0 |

## Failed Agents (0)

## Resource Usage
- Memory (RSS): 72.0 MB
- Memory (VMS): 35140.3 MB
- Memory percent: 0.8%
- CPU: 0.1%

> Note: GPU usage not available on this system.

## Bottlenecks & Recommendations
1. **Optimize slow agents** by profiling and parallelizing where possible
2. **Reduce retries** by fixing root causes of agent failures
3. **Consider incremental processing** for large datasets
4. **Add checkpoint recovery** to resume from failures
5. **Monitor memory** for large dataset ingestion, consider chunking
