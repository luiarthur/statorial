.PHONY: all

all:
	jupyter-nbconvert --to=python test.ipynb
	jupyter-nbconvert --to=html test.ipynb
	jupyter-nbconvert --to=markdown test.ipynb

clean:
	rm -f *.py *.html *.md
