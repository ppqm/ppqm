import os


def readlines_reverse(filename):
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


def enumerate_reversed(lines, max_lines=None, length=None):

    if length is None:
        length = len(lines)

    if max_lines is None:
        iterator = reversed(range(length))
    else:
        iterator = reversed(range(min(length, max_lines)))

    for index in iterator:
        yield index, lines[index]


def get_index(lines, pattern, stoppattern=None):

    for i, line in enumerate(lines):

        if pattern in line:
            return i

        if stoppattern and stoppattern in line:
            return None

    return None


def get_indices(lines, pattern, stoppattern=None):

    idxs = []

    for i, line in enumerate(lines):

        if pattern in line:
            idxs.append(i)

        if stoppattern and stoppattern in line:
            break

    return idxs


def get_indices_patterns(lines, patterns, stoppattern=None):

    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs = [None] * n_patterns

    for i, line in enumerate(lines):

        for ip in i_patterns:

            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

        if stoppattern and stoppattern in line:
            break

    return idxs


def get_rev_index(lines, pattern, stoppattern=None):

    for i, line in enumerate_reversed(lines):

        if pattern in line:
            return i

        if stoppattern and stoppattern in line:
            return None

    return None


def get_rev_indices_patterns(lines, patterns, stoppattern=None):

    n_patterns = len(patterns)
    i_patterns = list(range(n_patterns))

    idxs = [None] * n_patterns

    for i, line in enumerate_reversed(lines):

        if stoppattern and stoppattern in line:
            break

        for ip in i_patterns:

            pattern = patterns[ip]

            if pattern in line:
                idxs[ip] = i
                i_patterns.remove(ip)

    return idxs
