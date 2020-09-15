
import os


def readlines_reverse(filename):
    with open(filename) as qfile:
        qfile.seek(0, os.SEEK_END)
        position = qfile.tell()
        line = ''
        while position >= 0:
            qfile.seek(position)
            next_char = qfile.read(1)
            if next_char == "\n":
                yield line[::-1]
                line = ''
            else:
                line += next_char
            position -= 1
        yield line[::-1]


def get_index(
    lines,
    pattern,
    offset=None,
    n_lines=None,
    stoppattern=None
):

    if offset is None:
        offset = 0

    if n_lines is None:
        n_lines = len(lines)

    for i in range(offset, n_lines):
        line = lines[i]
        if line.find(pattern) != -1:
            return i

        if stoppattern and stoppattern in line:
            return None

    return None


def reverse_enum(L, max_lines=None, lenl=None):

    if lenl is None:
        lenl = len(L)

    if max_lines is None:
        iterator = reversed(range(lenl))
    else:
        iterator = reversed(range(min(lenl, max_lines)))

    for index in iterator:
        yield index, L[index]


def get_rev_index(
    lines,
    pattern,
    max_lines=None,
    lenl=None,
    stoppattern=False
):

    for i, line in reverse_enum(lines, max_lines=max_lines):

        if line.find(pattern) != -1:
            return i

        if stoppattern and stoppattern in line:
            return None

    return None


def get_indexes(lines, pattern):

    idxs = []

    for i, line in enumerate(lines):
        if pattern in line:
            idxs.append(i)

    return idxs


def get_indexes_with_stop(lines, pattern, stoppattern):

    idxs = []

    for i, line in enumerate(lines):
        if pattern in line:
            idxs.append(i)
            continue

        if stoppattern in line:
            break

    return idxs


def get_indexes_patterns(lines, patterns):

    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs = [None]*n_patterns

    for i, line in enumerate(lines):

        for ip in i_patterns:

            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

    return idxs


def get_rev_indexes(lines, patterns):

    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs = [None]*n_patterns

    for i, line in reverse_enum(lines):

        for ip in i_patterns:

            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

    return idxs
