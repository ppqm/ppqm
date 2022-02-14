import copy
import json
import logging
import multiprocessing
import pathlib
import pickle
import sys
import threading
from io import StringIO

import numpy as np
from tqdm import tqdm

try:
    import thread
except ImportError:
    import _thread as thread

from ppqm import constants

_logger = logging.getLogger("ppqm")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    return


def save_obj(name, obj):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def save_array(arr):
    s = StringIO()
    np.savetxt(s, arr)
    return s.getvalue()


def load_array(txt):
    s = StringIO(txt)
    arr = np.loadtxt(s)
    return arr


def str_json(dictionary, indent=4, translate_types=False):
    if translate_types:
        dictionary = json_friendly(dictionary)
    return json.dumps(dictionary, indent=indent)


def save_json(name, obj, indent=4, translate_types=False):
    """Save dictionary as a JSON file.

    translate_types: Change instances types of dictionary values to make it
    json friendly

    """

    if translate_types:
        obj = json_friendly(obj)

    if isinstance(name, str):
        name = pathlib.Path(name)

    if not name.suffix == ".json":
        name = name.with_suffix(".json")

    with open(name, "w") as f:
        json.dump(obj, f, indent=indent)


def json_friendly(dictionary):
    """ Change the types of a dictionary to make it JSON friendly """

    dictionary = copy.deepcopy(dictionary)

    for key, value in dictionary.items():

        if isinstance(value, dict):
            value = json_friendly(value)
            dictionary[key] = value

        elif isinstance(value, np.ndarray):
            value = value.tolist()
            dictionary[key] = value

    return dictionary


def load_json(name):

    if isinstance(name, str):
        name = pathlib.Path(name)

    if not name.suffix == ".json":
        name = name.with_suffix(".json")

    with open(name, "r") as f:
        content = f.read()
        content = json.loads(content)

    return content


def is_float(value, return_value=False):
    """ Return value as float, if possible """
    try:
        value = float(value)
    except ValueError:
        if not return_value:
            return False

    if return_value:
        return value

    return True


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


def meta_func(func, args, **kwargs):
    """ meta func to translate positional args to tuple """
    return func(*args, **kwargs)


def func_parallel(
    func,
    arg_list,
    show_progress=True,
    n_cores=1,
    n_jobs=None,
    title="Parallel",
):
    """
    Start pool of procs for function with arg_list.

    Use partial to set kwargs and non-variable args

    # TODO Finish example
    >>> def most_important(x, y, z, add=5):
    >>>     result = x + y + z + add
    >>>     return result
    >>>
    >>> x_list =
    >>>
    >>>


    """

    if show_progress:

        if n_jobs is None:
            n_jobs = len(arg_list)

        pbar = tqdm(
            total=n_jobs,
            desc=f"{title}({n_cores})",
            **constants.TQDM_OPTIONS,
        )

    p = multiprocessing.Pool(processes=n_cores)

    results = []

    try:
        results_iter = p.imap(func, arg_list, chunksize=1)

        for result in results_iter:

            if show_progress:
                pbar.update(1)

            results.append(result)

    except KeyboardInterrupt:
        eprint("got ^C while running pool of workers...")
        p.terminate()
        raise KeyboardInterrupt

    except Exception as e:
        eprint("got exception: %r, terminating the pool" % (e,))
        p.terminate()
        raise e

    finally:
        p.terminate()

    if show_progress:
        pbar.close()

    return results


def exit_after(sec):
    """
    use as decorator to exit process if function takes longer than s seconds

    usage:

    >>> @exit_after(5)
    >>> def countdown(n):
    >>>     i = 0
    >>>     while True:
    >>>         i += 1
    >>>         time.sleep(1)
    >>>         print(i)

    """

    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(sec, quit_function, args=[fn.__name__])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


def quit_function(fn_name):

    _logger.error(f"function '{fn_name}' took too long")

    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt
