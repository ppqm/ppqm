import logging
import os
import shutil
import subprocess
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Generator, Optional, Tuple, Union

_logger = logging.getLogger(__name__)


def _switch_workdir(path: Optional[Path]) -> bool:
    """Check if it makes sense to change directory"""

    if path is None:
        return False

    if path == Path("."):
        return False

    assert path.is_dir(), f"Cannot change directory, does not exists {path}"

    return True


def stream(cmd: str, cwd: Optional[Path] = None, shell: bool = True) -> Generator[str, None, None]:
    """Execute command in directory, and stream stdout. Last yield is
    stderr

    :param cmd: The shell command
    :param cwd: Change directory to work directory
    :param shell: Use shell or not in subprocess
    :param timeout: Stop the process at timeout (seconds)
    :returns: Generator of stdout lines. Last yield is stderr.
    """

    if _switch_workdir(cwd):
        cwd = None

    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
    )

    for stdout_line in iter(popen.stdout.readline, ""):  # type: ignore
        yield stdout_line

    # Yield errors
    stderr = popen.stderr.read()  # type: ignore
    popen.stdout.close()  # type: ignore
    yield stderr
    return


def execute(
    cmd: str, cwd: Optional[Path] = None, shell: bool = True, timeout: Optional[int] = None
) -> Tuple[str, str, bool]:
    """Execute command in directory, and return stdout and stderr

    :param cmd: The shell command
    :param cwd: Change directory to work directory
    :param shell: Use shell or not in subprocess
    :param timeout: Stop the process at timeout (seconds)
    :returns: stdout and stderr as string
    """

    if not _switch_workdir(cwd):
        cwd = None

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except TimeoutExpired:
        stdout = ""
        stderr = ""

    return_code = process.poll()

    return stdout, stderr, return_code


def source(bashfile: Path) -> dict:
    """
    Return resulting environment variables from sourceing a bashfile

    usage:
        env_dict = source("/path/to/aws_cred_ang")
        os.environ.update(env_dict)

    :returns: dict of variables
    """

    assert bashfile.is_file(), "File does not exist"

    cmd = f'env -i sh -c "source {bashfile} && env"'
    stdout, _ = execute(cmd)

    assert stdout is not None

    lines = stdout.split("\n")

    variables = dict()

    for line in lines:

        line_ = line.split("=")

        # Ignore wrong lines
        # - empty
        # - multiple =
        if len(line) != 2:
            continue

        key = line_[0]
        var = line_[1]

        if key == "PWD":
            continue

        if key == "_":
            continue

        if key == "SHLVL":
            continue

        variables[key] = var.strip()

    return variables


def which(cmd: str) -> Optional[str]:
    """find location of command in system"""
    return shutil.which(cmd)


def command_exists(cmd: str) -> bool:
    """does the command exists in current system?"""

    path = which(cmd)

    if path is None:
        return False

    return True


def get_threads() -> Optional[int]:

    n: Union[None, int, str] = os.environ.get("OMP_NUM_THREADS", None)

    if n is None:
        return None

    n = int(n)
    return n


def set_threads(n_cores: int) -> None:
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

    n_cores_ = str(n_cores)

    os.environ["OMP_NUM_THREADS"] = n_cores_
    os.environ["OPENBLAS_NUM_THREADS"] = n_cores_
    os.environ["MKL_NUM_THREADS"] = n_cores_
    os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores_
    os.environ["NUMEXPR_NUM_THREADS"] = n_cores_


def is_notebook() -> bool:
    """Check if module is called from a notebook enviroment"""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter
