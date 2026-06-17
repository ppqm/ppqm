import os
from collections.abc import Generator
from pathlib import Path


def readlines_reverse(filename: Path) -> Generator[str, None, None]:
    """Read file, line by line, backwards"""
    with open(filename) as qfile:
        qfile.seek(0, os.SEEK_END)
        position = qfile.tell()
        line = ""
        while position >= 0:
            qfile.seek(position)
            next_char = qfile.read(1)
            if next_char == "\n":
                yield line[::-1]
                line = ""
            else:
                line += next_char
            position -= 1
        yield line[::-1]


def enumerate_reversed(
    lines: list[str], max_lines: int | None = None, length: int | None = None
) -> Generator[tuple[int, str], None, None]:
    """Enumerate over list values, backwards"""

    if length is None:
        length = len(lines)

    if max_lines is None:
        iterator = reversed(range(length))
    else:
        iterator = reversed(range(min(length, max_lines)))

    for index in iterator:
        yield index, lines[index]


def get_index(
    lines: list[str],
    pattern: str,
    stoppattern: str | None = None,
    maxiter: int | None = None,
) -> int | None:
    for i, line in enumerate(lines):
        if pattern in line:
            return i

        if stoppattern and stoppattern in line:
            return None

        if maxiter and i > maxiter:
            return None

    return None


def get_indices(lines: list[str], pattern: str, stoppattern: str | None = None) -> list[int]:
    idxs = []

    for i, line in enumerate(lines):
        if pattern in line:
            idxs.append(i)

        if stoppattern and stoppattern in line:
            break

    return idxs


def get_rev_indices(lines: list[str], pattern: str, stoppattern: str | None = None) -> list[int]:
    idxs = []

    for i, line in enumerate_reversed(lines):
        if pattern in line:
            idxs.append(i)

        if stoppattern and stoppattern in line:
            break

    return idxs


def get_indices_patterns(
    lines: list[str], patterns: list[str], stoppattern: str | None = None
) -> list[int | None]:
    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs: list[int | None] = [None] * n_patterns

    for i, line in enumerate(lines):
        for ip in i_patterns:
            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

        if stoppattern and stoppattern in line:
            break

    return idxs


def get_indices_pattern(
    lines: list[str], pattern: str, num_lines: int, offset: int
) -> list[int | None]:
    """Processes the output file of the QM software used to
    find the first occurence of a specifie pattern. Useful
    if this block will be in the file only once and if there
    is no line that explicitly indicates the end of the block.

    Args:
        lines (list):
            Log file of the QM software to be processed.
        pattern (str):
            The pattern of the block.
        num_lines (int):
            How many line should be read.
        offset (int):
            How many lines are between the pattern and the first line of the block.

    Returns:
        list: Indices of the first and the last line of the block (including the offset).
    """

    idxs: list[int | None] = [None] * 2

    for i, line in enumerate(lines):
        if pattern in line:
            # only go with first occurence
            idxs[0] = i + offset
            idxs[1] = i + num_lines + offset
            break

    return idxs


def get_rev_index(lines: list[str], pattern: str, stoppattern: str | None = None) -> int | None:
    for i, line in enumerate_reversed(lines):
        if pattern in line:
            return i

        if stoppattern and stoppattern in line:
            return None

    return None


def get_rev_indices_patterns(
    lines: list[str],
    patterns: list[str],
    stoppattern: str | None = None,
    maxiter: int | None = None,
) -> list[int | None]:
    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs: list[int | None] = [None] * n_patterns

    # TODO Better way of admin how many lines are read
    n_read = 0

    for i, line in enumerate_reversed(lines):
        if stoppattern and stoppattern in line:
            break

        if maxiter and n_read > maxiter:
            break

        for ip in i_patterns:
            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

        n_read += 1

        continue

    return idxs
