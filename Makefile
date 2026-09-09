TEX = pdflatex -interaction=nonstopmode -halt-on-error

all: paper.pdf

paper.pdf: paper.tex fig/*.png
	$(TEX) paper.tex
	$(TEX) paper.tex
	@echo "pages: $$(pdfinfo paper.pdf | awk '/^Pages/{print $$2}')"

anon: paper_anon.tex fig/*.png
	$(TEX) paper_anon.tex
	$(TEX) paper_anon.tex
	@echo "pages: $$(pdfinfo paper_anon.pdf | awk '/^Pages/{print $$2}')"

figures:
	python scripts/cm.py
	python scripts/cm_dark.py

clean:
	rm -f *.aux *.log *.out *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all anon figures clean
