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


# Default scratch space
SCR = os.environ.get("TMPDIR", "./_tmp_test_")
SCR = pathlib.Path(SCR)

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
    "scr": SCR,
    "cmd": CONFIG["gamess"]["cmd"],
    "gamess_scr": CONFIG["gamess"]["scr"],
    "gamess_userscr": CONFIG["gamess"]["userscr"],
}

MOPAC_OPTIONS = {
    "scr": SCR,
    "cmd": CONFIG["mopac"]["cmd"],
}

MNDO_OPTIONS = {
    "scr": SCR,
    "cmd": CONFIG["mndo"]["cmd"],
}

XTB_OPTIONS = dict()
XTB_OPTIONS["scr"] = SCR

if "xtb" in CONFIG._sections:
    XTB_OPTIONS["cmd"] = CONFIG["xtb"]["cmd"]


G16_OPTIONS = dict()
G16_OPTIONS["scr"] = SCR
G16_OPTIONS["cmd"] = CONFIG["gaussian16"]["cmd"]

ORCA_OPTIONS = dict()
ORCA_OPTIONS["scr"] = SCR
ORCA_OPTIONS["cmd"] = CONFIG["orca"]["cmd"]
