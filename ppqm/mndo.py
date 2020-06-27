
import copy
from typing import Dict, List, Str

from calculator import CalculatorSkeleton


MNDO_CMD = "mndo"
MNDO_ATOMLINE = "{atom:2s} {x} {opt_flag} {y} {opt_flag} {z} {opt_flag}"


def MndoCalculator(CalculatorSkeleton):


    def __init__(self, cmd=MNDO_CMD, scr="./")

        self.cmd = cmd
        self.scr = scr

        # Ensure scrdir
        Path(scr).mkdir(parents=True, exist_ok=True)

        # Constants
        self.atomline = MNDO_ATOMLINE

        return


    def optimize(self, molobj,
        return_copy=True,
        return_properties=False):

        header = """{method} MULLIK PRECISE charge={charge} jprint=5\nnextmol=-1\nTITLE {title}"""

        if return_copy:
            molobj = copy.deepcopy(molobj)

        # TODO Get atoms, coord, charge
        # TODO Get input strings
        # TODO Set file
        # TODO Run mndo
        # TODO Parse output
        # TODO Set coord
        # TODO Set properties?

        return molobj


    def _set_input_file(self, input_str):

        # TODO Set in scr

        return


# Global functions

def set_input_str(atoms, coord, header, read_params=False, atomline=MNDO_ATOMLINE):
    """
    """

    # WARNING: INTERNAL COORDINATES ARE ASSUMED -
    # FOR THREE-ATOM SYSTEMS

    n_atoms = len(atoms)

    txt = header.format(method=method, charge=charge, title=title)

    if read_params:
        txt = txt.split("\n")
        txt[0] += " iparok=1"
        txt = "\n".join(txt)

    txt += "\n"

    if n_atoms <= 3:
        txt += internal_coordinates(atoms, coords, optimize=optimize)
        txt += "\n"
        return txt

    opt_flag = 0
    if optimize: opt_flag = 1

    for atom, coord in zip(atoms, coords):
        fmt = {
            "atom": atom,
            "x": coord[0],
            "x": coord[0],
            "x": coord[0],
            "opt_flag": opt_flag
        }
        line = atomline.format(**fmt)
        txt += line + "\n"

    txt += "\n"

    return


def set_internal_coordinates(
    atoms,
    coord,
    optimize=False):
    """

    :param atoms: List[Str]
    :param coord: Array[]
    :param optimize: Boolean
    :return Str:
    """

    n_atoms = len(atoms)

    opt_flag = 0
    if optimize: opt_flag = 1

    output = ""

    if (natoms == 3):

        ba = coord[1] - coord[0]
        bc = coord[1] - coord[2]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle) / np.pi * 180.0

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        output += f"{atoms[0]}\n"
        output += f"{atoms[1]} {norm_ba} {opt_flag}\n"
        output += f"{atoms[2]} {norm_bc} {opt_flag} {angle} {opt_flag}\n"

    elif (natoms == 2):

        ba = coord[1] - coord[0]
        norm_ba = np.linalg.norm(ba)
        output += f"{atoms[0]}\n"
        output += f"{atoms[1]} {norm_ba} {opt_flag}\n"

    elif (natoms == 1):

        output += f"{atoms[0]}\n"

    return output

