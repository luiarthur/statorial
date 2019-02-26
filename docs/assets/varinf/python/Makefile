PATH_TO_VARINF_PY = ../../../_varinf/python/

.PHONY: all linreg clean move

all: linreg move

move:
	sed -i 's/LinReg_files/{{site.baseurl}}{% link \/assets\/varinf\/python\/LinReg_files/g' LinReg.md
	sed -i 's/.png)/.png %})/g' LinReg.md
	mv LinReg.md $(PATH_TO_VARINF_PY)

linreg:
	jupyter-nbconvert --to=markdown LinReg.ipynb

clean:
	rm -rf LinReg_files LinReg.md
