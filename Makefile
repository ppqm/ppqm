python = python
LINE_LENGTH=79
BLACKARGS=--line-length ${LINE_LENGTH}

FLAKEARGS=\
	--remove-all-unused-imports \
	--remove-unused-variables \
	--expand-star-imports

src=ppqm/*.py tests/*.py

lint:
	${python} -m isort --check-only ${src}
	${python} -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	${python} -m flake8 . --count --exit-zero --max-complexity=10 --statistics
	${python} -m autoflake --check ${FLAKEARGS} ${src}
	${python} -m black --check ${BLACKARGS} ${src}

format:
	${python} -m isort ${src}
	${python} -m autoflake --in-place ${FLAKEARGS} ${src}
	${python} -m black ${BLACKARGS} ${src}

test:
	${python} -m pytest -vrs tests
