.PHONY: install lint test smoke data
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
lint:
	ruff check .
test:
	pytest -q
data:
	python scripts/download_data.py
smoke:
	python scripts/local_smoke_test.py
