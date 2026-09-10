PYTHON ?= python

.PHONY: install test lint split results manifest verify paper-figures paper-claims paper-tables paper paper-check clean-paper

install:
	$(PYTHON) -m pip install -e '.[dev,tables,paper]'

test:
	pytest -q

lint:
	ruff check src tests scripts

split:
	$(PYTHON) data_construction/verify_split.py

results:
	$(PYTHON) results/build_results_tables.py
	$(PYTHON) scripts/verify_authoritative_results.py

manifest:
	$(PYTHON) scripts/build_release_manifest.py --verify

verify: split results test lint manifest

paper-figures:
	$(PYTHON) scripts/vectorize_figure_assets.py
	$(PYTHON) scripts/render_paper_figures.py

paper-claims:
	$(PYTHON) scripts/build_paper_claim_ledger.py

paper-tables:
	$(PYTHON) scripts/render_paper_tables.py

paper: paper-figures paper-claims paper-tables
	cd paper && SOURCE_DATE_EPOCH=1788955200 FORCE_SOURCE_DATE=1 TEXINPUTS=template//: BSTINPUTS=template//: BIBINPUTS=.: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cd paper && SOURCE_DATE_EPOCH=1788955200 FORCE_SOURCE_DATE=1 TEXINPUTS=template//: BSTINPUTS=template//: BIBINPUTS=.: latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex

paper-check: paper
	$(PYTHON) scripts/verify_paper_references.py
	$(PYTHON) scripts/verify_paper_artifacts.py

clean-paper:
	cd paper && latexmk -C main.tex
	cd paper && latexmk -C supplementary.tex
