import os
import shutil
import subprocess


def check_cmd(cmd):
    """ Does this command even exists? """

    path = shutil.which(cmd)

    if path is None:
        return False

    return True


def check_path(path):
    """ Check if it makes sense to change directory """

    if path is None:
        return False

    if path == "":
        return False

    if path == "./":
        return False

    if path == ".":
        return False

    assert os.path.exists(path), f"Cannot chdir, directory does not exists {path}"

    return True


def stream(cmd, chdir=None, shell=True):
    """ Execute command in chdir directory, and stream stdout. Last yield is stderr """

    # TODO Is there a better way to run command from directory
    if check_path(chdir):
        cmd = f"cd {chdir}; " + cmd

    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
    )

    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    # Yield errors
    stderr = popen.stderr.read()
    popen.stdout.close()
    yield stderr

    return


def execute(cmd, chdir=None, shell=True):
    """ Execute command in chdir directory, and return stdout and stderr """

    if check_path(chdir):
        cmd = f"cd {chdir}; " + cmd

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)

    stdout, stderr = p.communicate()

    return stdout.decode(), stderr.decode()
