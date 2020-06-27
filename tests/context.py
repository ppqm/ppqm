
import sys
import os
import pathlib

parent = str(pathlib.Path(__file__).absolute().parent.parent)
sys.path.insert(0, parent)

import ppqm

# TODO Set config for testing
# def ini_settings(filename):
#     config = configparser.ConfigParser()
#     config.read(filename)
#     return config
#
# # Init enviroment
# CONFIG = ini_settings("development.ini")
# Path(SCR).mkdir(parents=True, exist_ok=True)

