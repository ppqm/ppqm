python=python
mamba=mamba
pkg=ppqm
pip=./env/bin/pip

all: env

env_minimal:
	${mamba} env create -f ./environment_minimal.yml -p ./env --quiet
	${pip} install -e .

env:
	${mamba} env create -f ./environment_interactive.yml -p ./env --quiet
	${pip} install -e .

setup-dev:
	pre-commit install

format:
	pre-commit run --all-files

test:
	${python} -m pytest -rs tests

cov:
	${python} -m pytest -vrs --cov=${pkg} --cov-report html tests

diff-report:
	git diff '@{2 month ago}' HEAD > change_month.diff
	grep "smiles =" tests/*py > molecules.txt
	ls "tests/resources/compounds/" >> molecules.txt
