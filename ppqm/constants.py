import sys

from ppqm import env

# Default scratch directory
SCR = "./"

# Coordinates name
COLUMN_COORDINATES = "coords"
COLUMN_ATOMS = "atoms"
COLUMN_ENERGY = "total_energy"

COLUMN_ENTHALPY = ""


# tqdm default view
TQDM_OPTIONS = {
    "ncols": 80,
}

# if run from jupyter, print to stdout and not stderr
if env.is_notebook():
    TQDM_OPTIONS["file"] = sys.stdout
