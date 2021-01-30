import configparser
import logging
import os
import pathlib

# Set default logging to debug for all tests
logging.basicConfig(level=logging.DEBUG)


def read_settings(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config


SCR = os.environ.get("TMPDIR", "./_tmp_test_")
SCR = pathlib.Path(SCR)
SCR.mkdir(parents=True, exist_ok=True)

# Resources path
RESOURCES = pathlib.Path("tests/resources")

# Try to read config files
configfile = pathlib.Path("development.ini")

if not configfile.is_file():
    configfile = pathlib.Path("production.ini")

if not configfile.is_file():
    configfile = pathlib.Path("default.ini")

CONFIG = read_settings(configfile)

# gamess standard
GAMESS_OPTIONS = {
    "scr": CONFIG["scr"]["scr"],
    "cmd": CONFIG["gamess"]["cmd"],
    "gamess_scr": CONFIG["gamess"]["scr"],
    "gamess_userscr": CONFIG["gamess"]["userscr"],
}
