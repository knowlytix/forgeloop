# forgeloop — one entry point for the things a reader or contributor runs.
#
# The repo holds four installable packages: one per book plus the assembled
# `forgeloop` distribution. Installing them by hand means four commands in the
# right order, and getting the order wrong fails in a confusing way -- Ship-and
# -Pray's SUT imports agentlab, which is not declared as a dependency because
# agentlab is not published to PyPI. `make install` does it correctly.

PYTHON ?= python3
PIP    := $(PYTHON) -m pip

.PHONY: help install install-dev assemble check lint test docs clean

help:
	@echo "make install      everything, ready to run"
	@echo "make install-dev  install plus test and lint extras"
	@echo "make assemble     rebuild forgeloop/ and the docs gallery from the books"
	@echo "make check        everything CI runs: lint, notebook guard, drift, tests"
	@echo "make docs         build the documentation site into docs/docs/_build/html"

# Order matters: agentlab first, because beyond-ship-and-pray imports it.
#
# This installs the GMS extras too, so every book works after one command. It is
# a few GB, mostly torch and transformers. Chunk-and-Pray in particular is not
# usable without them -- 27 of its 40 notebooks load the substrate and only 4
# are plain Python -- so making its readers run a second command to get anything
# working would be the wrong default. Use `make install-lite` to skip them.
install:
	$(PIP) install -e "beyond-prompt-and-pray/code[gms,ml,notebooks]"
	$(PIP) install -e "beyond-ship-and-pray/code[gms,notebooks]"
	$(PIP) install -e "beyond-chunk-and-pray/code[gms,ml,notebooks]"
	$(PIP) install -e .
	@echo
	@echo "installed. knowlytix needs a licence key to run (installing it is free):"
	@echo "    ~/.knowlytix/license.key    https://knowlytix.ai/signup/"


install-dev:
	$(PIP) install -e "beyond-prompt-and-pray/code[dev]"
	$(PIP) install -e "beyond-ship-and-pray/code[dev]"
	$(PIP) install -e "beyond-chunk-and-pray/code[dev]"
	$(PIP) install -e ".[dev]"


assemble:
	$(PYTHON) scripts/build_forgeloop.py

# Mirrors the CI jobs, so a green run here means a green run there.
check: lint
	$(PYTHON) scripts/notebook_outputs.py --check
	$(PYTHON) scripts/build_forgeloop.py --package --check
	$(MAKE) test

lint:
	ruff check .

test:
	$(PYTHON) -m pytest -q \
	    beyond-prompt-and-pray/code/tests \
	    beyond-ship-and-pray/code/tests

docs:
	$(MAKE) -C docs/docs html
	@echo "open docs/docs/_build/html/index.html"

clean:
	$(MAKE) -C docs/docs clean 2>/dev/null || true
	rm -rf docs/4-notebook-examples
