.PHONY: install test lint split results manifest verify paper-figures paper-claims paper-tables paper paper-check clean-paper

install:
	python -m pip install -e '.[dev,tables]'

test:
	pytest -q

lint:
	ruff check src tests scripts

split:
	python data_construction/verify_split.py

results:
	python results/build_results_tables.py
	python scripts/verify_authoritative_results.py

manifest:
	python scripts/build_release_manifest.py --verify

verify: split results test lint manifest

paper-figures:
	python scripts/render_paper_figures.py

paper-claims:
	python scripts/build_paper_claim_ledger.py

paper-tables:
	python scripts/render_paper_tables.py

paper: paper-figures paper-claims paper-tables
	cd paper && SOURCE_DATE_EPOCH=1787632110 FORCE_SOURCE_DATE=1 TEXINPUTS=template//: BSTINPUTS=template//: BIBINPUTS=.: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cd paper && SOURCE_DATE_EPOCH=1787632110 FORCE_SOURCE_DATE=1 TEXINPUTS=template//: BSTINPUTS=template//: BIBINPUTS=.: latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex

paper-check: paper
	python scripts/verify_paper_references.py
	python scripts/verify_paper_artifacts.py

clean-paper:
	cd paper && latexmk -C main.tex
	cd paper && latexmk -C supplementary.tex
