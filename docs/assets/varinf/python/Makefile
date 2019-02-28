PATH_TO_VARINF_PY = ../../../_varinf/python/

.PHONY: all linreg clean move

all: linreg move

move:
	mv LinReg.html $(PATH_TO_VARINF_PY)

linreg:
	jupyter-nbconvert LinReg.ipynb --to=html --template=basic \
		--HTMLExporter.anchor_link_text=' ' \
		--TemplateExporter.exclude_input_prompt=True
