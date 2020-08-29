
import subprocess
import os
import shutil


def check_cmd(cmd):

    path = shutil.which(cmd)

    if path is None:
        return False

    return True


def check_path(path):
    """

    """

    if path is None:
        return False

    if path == "":
        return False

    if path == "./":
        return False

    assert os.path.exists(path), f"Cannot chdir, directory does not exists {path}"

    return True


def stream(cmd, chdir=None):
    """

    :param cmd:
    """

    if check_path(chdir):
        cmd = f"cd {chdir}; " + cmd

    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        shell=True
    )

    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()

    # TODO also yield stderr

    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def execute(cmd, chdir=None):
    """
    """

    if check_path(chdir):
        cmd = f"cd {chdir}; " + cmd

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )

    stdout, stderr = p.communicate()

    return stdout.decode(), stderr.decode()
