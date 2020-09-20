
import numpy as np
from io import StringIO
import sys
import subprocess
import pickle


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    return


def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_array(arr):
    s = StringIO()
    np.savetxt(s, arr)
    return s.getvalue()


def load_array(txt):
    s = StringIO(txt)
    arr = np.loadtxt(s)
    return arr


def merge_dict(a, b, path=None):
    "merges dictionary b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a
