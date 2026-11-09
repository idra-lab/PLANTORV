PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
RUFF ?= ruff
PYRIGHT ?= pyright
PRE_COMMIT ?= pre-commit
SPHINX_BUILD ?= sphinx-build
SPHINX_APIDOC ?= sphinx-apidoc
SPHINX_AUTOBUILD ?= sphinx-autobuild
O3DML_VENV ?= .venv-o3dml

# Packages the API documentation covers. sphinx-apidoc is run once per package: pointing
# it at the project root instead would make the root itself a namespace package and
# prefix every module with `plantorv.`, which is not an importable name.
DOC_PACKAGES ?= segmentation scene_understanding mapping LLM utility aruco evaluation pipeline

.PHONY: install install-dev install-o3dml format format-check lint typecheck check docs docs-api docs-serve

# Runtime dependencies only - this is what the PBS cluster jobs need.
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# The Open3D-ML environment, in a virtualenv of its own. Open3D 0.19 only loads its
# PyTorch ops under torch 2.2.*, which the main environment left behind for SAM 3, so
# the learned point-cloud models of segment_pcd.py live here instead. See
# requirements-o3dml.txt.
install-o3dml:
	$(PYTHON) -m venv $(O3DML_VENV)
	$(O3DML_VENV)/bin/pip install --upgrade pip
	$(O3DML_VENV)/bin/pip install -r requirements-o3dml.txt
	@echo "Run the learned point-cloud models with $(O3DML_VENV)/bin/python segment_pcd.py"

# Runtime + dev tooling (ruff, pyright, pre-commit), plus the git hook.
# pip has no way to pull an extra automatically, so `make install-dev` is the
# single entry point instead of remembering `pip install -e ".[dev]"`.
install-dev: install
	$(PIP) install -e ".[dev]"
	$(PRE_COMMIT) install

format:
	$(RUFF) check --fix .
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

# Build the API documentation into docs/_build/html. Depends on docs-api, so the stubs are
# generated rather than committed: they derive entirely from the package layout, and a stale
# checked-in copy would silently document the wrong set of modules.
docs: docs-api
	$(SPHINX_BUILD) -b html docs docs/_build/html

# Serve the API documentation at http://localhost:8000, rebuilding on changes.
docs-serve: docs-api
	$(SPHINX_AUTOBUILD) docs docs/_build/html

# Write the per-module stubs under docs/api/, one `automodule` directive each. Runs as part
# of `make docs`; useful on its own only to inspect what it generates.
docs-api:
	for pkg in $(DOC_PACKAGES); do \
		$(SPHINX_APIDOC) -f -e -M --implicit-namespaces -o docs/api $$pkg \
			"$$pkg/conf" "$$pkg/examples" "$$pkg/__pycache__" "$$pkg/run_evaluation_old.py"; \
	done
	rm -f docs/api/modules.rst

clean: clean_ruffy clean_pyright clean_output clean_package clean_docs

clean_ruffy:
	if [ -d .ruff_cache ]; then rm -rf .ruff_cache; fi
clean_pyright:
	if [ -d .pyrightcache ]; then rm -rf .pyrightcache; fi
clean_output: 
	rm -rf output
	rm -rf results
	rm -f ./*.log
clean_docs:
	if [ -d docs/_build ]; then rm -rf docs/_build; fi
	if [ -d docs/api ]; then rm -rf docs/api; fi
clean_package:
	if [ -d dist ]; then rm -rf dist; fi
	if [ -d build ]; then rm -rf build; fi
	if [ -d plantorv.egg-info ]; then rm -rf plantorv.egg-info; fi