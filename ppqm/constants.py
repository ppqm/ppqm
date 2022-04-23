import sys
from pathlib import Path
from typing import Any, Dict

from ppqm.utils import shell

# Default scratch directory
SCR = Path("./")

# Coordinates name
COLUMN_COORDINATES = "coords"
COLUMN_ATOMS = "atoms"
COLUMN_ENERGY = "total_energy"

COLUMN_ENTHALPY = ""


# tqdm default view
TQDM_OPTIONS: Dict[str, Any] = {
    "ncols": 80,
}

# if run from jupyter, print to stdout and not stderr
if shell.is_notebook():
    TQDM_OPTIONS["file"] = sys.stdout
