import configparser
import pathlib
import sys

try:
    import ppqm

except ImportError:
    parent = str(pathlib.Path(__file__).absolute().parent.parent)
    sys.path.insert(0, parent)
    import ppqm


def read_settings(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config


configfile = pathlib.Path("development.ini")

if not configfile.is_file():
    configfile = pathlib.Path("production.ini")

if not configfile.is_file():
    configfile = pathlib.Path("default.ini")

CONFIG = read_settings(configfile)
