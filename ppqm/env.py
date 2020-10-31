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
