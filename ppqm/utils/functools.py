import logging
import multiprocessing
import sys
import threading
from typing import Any, Callable, List, Optional, Sequence

from tqdm import tqdm  # type: ignore

try:
    import thread  # type: ignore
except ImportError:
    import _thread as thread  # type: ignore[no-redef]

from ppqm import constants

_logger = logging.getLogger(__name__)


def func_parallel(
    func: Callable,
    arg_list: Sequence,
    show_progress: bool = True,
    n_cores: int = 1,
    n_jobs: Optional[int] = None,
    title: str = "Parallel",
) -> List[Any]:
    """
    Start pool of procs for function with arg_list.

    Use partial to set kwargs and non-variable args

    Example:
    >>> def most_important(x, y, z, add=5):
    >>>     result = x + y + z + add
    >>>     return result
    >>> x_list = [5, 7, 5, 3]
    >>> func = functools.partial(most_important, y, z, add=7)
    >>> results = func_parallel(func, x_list, n_cores=1, n_jobs=len(x_list))

    """

    pbar = None

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

            if pbar:
                pbar.update(1)

            results.append(result)

    except KeyboardInterrupt:
        _logger.error("got ^C while running pool of workers...")
        p.terminate()
        raise KeyboardInterrupt

    except Exception as e:
        _logger.error(f"got exception: {e}, terminating the pool")
        p.terminate()
        raise e

    finally:
        p.terminate()

    if pbar:
        pbar.close()

    return results


# TODO What todo with typing for a decorator?
def exit_after(sec: int) -> Any:
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

    def outer(fn: Callable) -> Any:
        def inner(*args, **kwargs):  # type: ignore
            timer = threading.Timer(sec, quit_function, args=[fn])
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result

        return inner

    return outer


def quit_function(func: Callable, reason: str = "took too long") -> None:
    """Raise KeyboardInterrupt"""

    _logger.error(f"function '{func.__name__}' quit, because {reason}")

    sys.stderr.flush()  # Python 3 stderr is likely buffered.
    thread.interrupt_main()  # raises KeyboardInterrupt
