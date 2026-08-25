.PHONY: install test lint split results manifest verify paper-figures paper clean-paper

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

paper: paper-figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/supplementary.tex

clean-paper:
	latexmk -C -cd paper/main.tex
	latexmk -C -cd paper/supplementary.tex
