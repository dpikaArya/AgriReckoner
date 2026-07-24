# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Continuous integration workflow (`.github/workflows/ci.yml`): lint, a
  Python 3.10/3.12 test matrix, and a wheel-smoke build.
- Developer tooling: `Makefile`, `.pre-commit-config.yaml`, and shared pytest
  fixtures in `tests/conftest.py`.
- This changelog.

### Changed
- Quality-uplift refactor in progress: aligning the codebase with the project
  engineering standards (naming, function size, docstrings, tests, and
  linting). No behavioural changes intended during this phase.

## [2.0.0]

- Baseline release of the Agricultural Intelligence Framework (AAIF).
