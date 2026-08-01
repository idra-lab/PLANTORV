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

# Format the code in place.
format:
	$(RUFF) format .

# Check that the code is formatted correctly, without changing it.
format-check:
	$(RUFF) format --check .

# Lint the code for style and correctness issues.
lint:
	$(RUFF) check .

# Typecheck the code for type errors.
typecheck:
	$(PYRIGHT)

check: format-check lint typecheck

clean: clean_ruffy clean_pyright clean_output clean_package

clean_ruffy:
	if [ -d .ruff_cache ]; then rm -rf .ruff_cache; fi
clean_pyright:
	if [ -d .pyrightcache ]; then rm -rf .pyrightcache; fi
clean_output: 
	if [ -d output ]; then rm -rf output; fi
	if [ -d ppt_outputs ]; then rm -rf ppt_outputs; fi
	if [ -d outputs_json_labeled ]; then rm -rf outputs_json_labeled; fi
	rm -f ./*.log
clean_package:
	if [ -d dist ]; then rm -rf dist; fi
	if [ -d build ]; then rm -rf build; fi
	if [ -d plantorv.egg-info ]; then rm -rf plantorv.egg-info; fi