python=python
mamba=mamba
pkg=ppqm


env:
	${mamba} env create -f ./environment.yml -p ./env

env_interactive:
	${mamba} env create -f ./environment_interactive.yml -p ./env

setup-dev:
	pre-commit install

test:
	${python} -m pytest -vrs tests

cov:
	${python} -m pytest -vrs --cov=${pkg} --cov-report html tests

diff-report:
	git diff '@{2 month ago}' HEAD > change_month.diff
	grep "smiles =" tests/*py > molecules.txt
	ls "tests/resources/compounds/" >> molecules.txt
