import os
import shutil


def which(cmd):
    """ find location of command in system """
    return shutil.which(cmd)


def command_exists(cmd):
    """ does the command exists in current system? """

    path = which(cmd)

    if path is None:
        return False

    return True


def get_threads():
    """"""

    n = os.environ.get("OMP_NUM_THREADS", None)

    if n is not None:
        n = int(n)

    return n


def set_threads(n_cores):
    """

    Wrapper for setting environmental variables related to threads and procs.

    export OMP_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export VECLIB_MAXIMUM_THREADS=4
    export MKL_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4

    :param n_cores: Number of threads to be used internally in compiled
    programs and libs.

    """

    n_cores = str(n_cores)

    os.environ["OMP_NUM_THREADS"] = n_cores
    os.environ["OPENBLAS_NUM_THREADS"] = n_cores
    os.environ["MKL_NUM_THREADS"] = n_cores
    os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores
    os.environ["NUMEXPR_NUM_THREADS"] = n_cores

    return


def is_notebook():
    """
    Check if module is called from a notebook enviroment
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
