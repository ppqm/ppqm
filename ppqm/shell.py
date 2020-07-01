
import subprocess


def stream(cmd, chdir=None):
    """

    :param cmd:
    """

    if chdir is not None:
        cmd = f"cd {chdir}; " + cmd

    popen = subprocess.Popen(cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        shell=True)

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

    p = subprocess.Popen(cmd,
        stdout=subprocess.PIPE,
        shell=True)

    stdout, stderr = p.communicate()

    return stdout, stderr

