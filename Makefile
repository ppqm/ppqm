python="python"

setup-dev:
	pre-commit install

test:
	${python} -m pytest -vrs tests

diff-report:
	git diff '@{2 month ago}' HEAD > change_month.diff
	grep "smiles =" tests/*py > molecules.txt
	ls "tests/resources/compounds/" >> molecules.txt
