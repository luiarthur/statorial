PATH_TO_VARINF_PY = ../../../_varinf/python/

.PHONY: all linreg clean move

all: linreg move

move:
	mv LinReg.html $(PATH_TO_VARINF_PY)

linreg:
	jupyter-nbconvert LinReg.ipynb --to=html --template=basic --config=nb_config.py
