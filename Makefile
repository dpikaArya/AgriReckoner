.PHONY: install test lint fmt cov ci

install:
	pip install -e .[dev,llm]

test:
	pytest -m 'not live'

lint:
	ruff check .

fmt:
	ruff format .

cov:
	pytest --cov=agri_ai_agent

ci: lint test
