python="python"

setup-dev:
	pre-commit install


test:
	${python} -m pytest -vrs tests
