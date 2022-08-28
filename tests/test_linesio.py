from ppqm.utils import linesio

EXAMPLE_LINES = [
    "alpha-1",
    "not this one",
    "omega-1",
    "alpha-2",
    "couple",
    "primer",
    "psi",
    "phi",
    "end",
]


def test_get_index() -> None:
    assert linesio.get_index(EXAMPLE_LINES, "alpha") == 0
    assert linesio.get_index(EXAMPLE_LINES, "psi") == 6
    assert linesio.get_index(EXAMPLE_LINES, "end", stoppattern="phi") is None


def test_get_indices() -> None:
    pattern = "alpha"
    indicies = linesio.get_indices(EXAMPLE_LINES, pattern, stoppattern="phi")
    assert indicies == [0, 3]


def test_get_indices_patterns() -> None:
    patterns = ["alpha", "omega", "end"]
    indicies = linesio.get_indices_patterns(EXAMPLE_LINES, patterns, stoppattern="phi")
    assert indicies == [0, 2, None]


def test_get_rev_index() -> None:
    assert linesio.get_rev_index(EXAMPLE_LINES, "phi") == 7
    assert linesio.get_rev_index(EXAMPLE_LINES, "alpha", stoppattern="primer") is None


def test_get_rev_indices_patterns() -> None:
    patterns = ["alpha", "end", "psi"]
    indicies = linesio.get_rev_indices_patterns(EXAMPLE_LINES, patterns, stoppattern="primer")
    assert indicies == [None, 8, 6]
