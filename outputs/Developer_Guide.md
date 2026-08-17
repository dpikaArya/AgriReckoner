# Developer Guide
Generated: 2026-07-07T22:01:32.428351

## Setup
```bash
pip install -r requirements.txt
python -m ades run input.xlsx
```

## Adding a New Agent
1. Create agent file in `ades/agents/agentXX_name.py`
2. Add agent class extending `BaseAgent`
3. Add contract in `ades/contracts/messages.py`
4. Register in `ades/agents/__init__.py`
5. Add step to `PIPELINE_STEPS` in `ades/orchestrator.py`

## Running Tests
```bash
pytest tests/ -v
```

## Running Evaluation
```bash
python -m evaluation.run_evaluation
```

## Output Structure
- `outputs/`: All pipeline outputs
- `evaluation/reports/`: Evaluation reports and dashboard
- `benchmarks/`: Gold standard benchmark data