import logging
import os
import shutil
import subprocess
from subprocess import TimeoutExpired

_logger = logging.getLogger("ppqm.sh")


def command_exists(cmd):
    """ Does this command even exists? """

    path = shutil.which(cmd)

    if path is None:
        return False

    return True


def switch_workdir(path):
    """ Check if it makes sense to change directory """

    if path is None:
        return False

    if path == "":
        return False

    if path == "./":
        return False

    if path == ".":
        return False

    assert os.path.exists(path), f"Cannot change directory, does not exists {path}"

    return True


def stream(cmd, cwd=None, shell=True):
    """Execute command in directory, and stream stdout. Last yield is
    stderr

    :param cmd: The shell command
    :param cwd: Change directory to work directory
    :param shell: Use shell or not in subprocess
    :param timeout: Stop the process at timeout (seconds)
    :returns: Generator of stdout lines. Last yield is stderr.
    """

    if not switch_workdir(cwd):
        cwd = None

    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
    )

    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    # Yield errors
    stderr = popen.stderr.read()
    popen.stdout.close()
    yield stderr

    return


def execute(cmd, cwd=None, shell=True, timeout=None):
    """Execute command in directory, and return stdout and stderr

    :param cmd: The shell command
    :param cwd: Change directory to work directory
    :param shell: Use shell or not in subprocess
    :param timeout: Stop the process at timeout (seconds)
    :returns: stdout and stderr as string
    """

    if not switch_workdir(cwd):
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
        stdout = None
        stderr = None

    return stdout, stderr


def source(bashfile):
    """
    Return resulting environment variables from sourceing a bashfile

    usage:
        env_dict = source("/path/to/aws_cred_ang")
        os.environ.update(env_dict)

    :returns: dict of variables
    """

    cmd = f'env -i sh -c "source {bashfile} && env"'
    stdout, stderr = execute(cmd)
    lines = stdout.split("\n")

    variables = dict()

    for line in lines:

        line = line.split("=")

        # Ignore wrong lines
        # - empty
        # - multiple =
        if len(line) != 2:
            continue

        key, var = line

        if key == "PWD":
            continue

        if key == "_":
            continue

        if key == "SHLVL":
            continue

        variables[key] = var

    return variables
