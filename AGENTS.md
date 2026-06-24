# Repository Guidelines

## Project Structure & Module Organization
- `src/ir/`: core IR modules (`text/`, `index/`, `retrieval/`, `eval/`, `ranking/`, `cluster/`, `summarize/`).
- `scripts/`: CLI tools for indexing/search/eval; `app.py` is the Flask demo, with UI in `templates/` and `static/`.
- `tests/`: pytest suite; `configs/` for YAML; `datasets/` and `data/` for sample corpora; `docs/` for documentation.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs all runtime, test, and dev tools.
- `pytest tests/` runs the suite with coverage and reports to `htmlcov/`.
- `pytest -m "not slow"` runs fast tests only.
- `python app.py` starts the web app at `http://localhost:5001`.
- `python scripts/boolean_search.py --query "information AND retrieval"` runs a CLI search.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, PEP 8; modules/files use `snake_case`, classes use `CapWords`, tests use `test_*.py`.
- Formatting and linting: `black src/ tests/ scripts/`, `flake8 src/ tests/`, `pylint src/ir/`, `mypy src/`.
- Code comments and docstrings are English-only; new functions/classes should note time/space complexity.
- Documentation prefers Traditional Chinese with bilingual technical terms; avoid duplicate docs and update `docs/CHANGELOG.md` when changing behavior or docs.

## Testing Guidelines
- Tests live in `tests/` and follow `test_*.py`, `Test*`, `test_*` naming.
- Use markers from `pytest.ini` (`unit`, `integration`, `slow`, `requires_data`, etc.) and keep coverage above the 80% threshold.
- Add at least normal, boundary, and edge-case coverage; prefer `datasets/mini/` for quick-running data tests.

## Commit & Pull Request Guidelines
- Use Conventional Commits like `feat:`, `fix:`, `docs:`, `test:`, `chore:` (e.g., `feat: add rocchio expansion CLI`).
- PRs should include a short summary, rationale, and how you tested; link related issues and include screenshots for UI changes in `templates/` or `static/`.
