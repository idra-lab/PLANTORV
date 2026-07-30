PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
RUFF ?= ruff
PYRIGHT ?= pyright
PRE_COMMIT ?= pre-commit

.PHONY: install install-dev format format-check lint typecheck check

# Runtime dependencies only - this is what the PBS cluster jobs need.
install:
	$(PIP) install -e .

# Runtime + dev tooling (ruff, pyright, pre-commit), plus the git hook.
# pip has no way to pull an extra automatically, so `make install-dev` is the
# single entry point instead of remembering `pip install -e ".[dev]"`.
install-dev:
	$(PIP) install -e ".[dev]"
	$(PRE_COMMIT) install

format:
	$(RUFF) format .

format-check:
	$(RUFF) format --check .

lint:
	$(RUFF) check .

typecheck:
	$(PYRIGHT)

check: format-check lint typecheck