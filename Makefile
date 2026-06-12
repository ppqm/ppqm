.PHONY: all env env_minimal format test cov build test-dist upload

env=env
python=${env}/bin/python
pkg=ppqm

all: env

env: ${env}_uv

${env}_uv:
	uv venv ${env}
	uv pip install -e . --python ${env}/bin/python
	uv pip install -e .[dev,test] --python ${env}/bin/python
	${python} -m pre_commit install

env_minimal: ${env}_uv_minimal

${env}_uv_minimal:
	uv venv ${env}
	uv pip install -e . --python ${env}/bin/python
	uv pip install -e .[test] --python ${env}/bin/python

setup-dev:
	pre-commit install

format:
	${python} -m pre_commit run --all-files

test:
	${python} -m pytest -rs tests

cov:
	${python} -m pytest -vrs --cov=${pkg} --cov-report html tests

build:
	${python} -m build --skip-dependency-check  .

test-dist:
	${python} -m twine check dist/*

upload:
	${python} -m twine upload ./dist/*

diff-report:
	git diff '@{2 month ago}' HEAD > change_month.diff
	grep "smiles =" tests/*py > molecules.txt
	ls "tests/resources/compounds/" >> molecules.txt
