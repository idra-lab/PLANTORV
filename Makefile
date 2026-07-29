PYTHON ?= python3
RUFF ?= ruff
PYRIGHT ?= pyright

.PHONY: format format-check lint typecheck check test

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

lint:
	$(RUFF) check .

typecheck:
	$(PYRIGHT)

check: format-check lint typecheck

test:
	$(PYTHON) -m unittest tests.test_segmentation tests.test_depth tests.test_vlm
