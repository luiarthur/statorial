.PHONY: all linreg clean

all: linreg

linreg:
	jupyter-nbconvert --to=python LinReg.ipynb
	jupyter-nbconvert --to=html LinReg.ipynb
	jupyter-nbconvert --to=markdown LinReg.ipynb

clean:
	rm -rf LinReg_files
	rm -f LinReg.html LinReg.md LinReg.py
