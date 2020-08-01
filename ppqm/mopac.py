
import os
import copy
from pathlib import Path
import typing
import numpy as np


from .calculator import CalculatorSkeleton
from . import chembridge
from . import misc
from . import shell
from . import linesio

# NOTE
# Should be possible to get graph of molecule using
# keyword "local" on mopac
# http://openmopac.net/manual/localize.html

# NOTE
# There is an internal check for distances, but can be ignored with
# "GEO-OK" keywords.


MOPAC_METHOD = "PM6"
MOPAC_VALID_METHODS = ["PM3", "PM6", "PM7"]
MOPAC_CMD = "mopac"
MOPAC_ATOMLINE = "{atom:2s} {x} {opt_flag} {y} {opt_flag} {z} {opt_flag}"
MOPAC_INPUT_EXTENSION = ".mop"
MOPAC_OUTPUT_EXTENSION = ".out"


class MopacCalculator(CalculatorSkeleton):
    """
    """

    def __init__(self,
        cmd=MOPAC_CMD,
        method=MOPAC_METHOD,
        filename="_tmp_mopac.mop",
        scr="./"):
        """
        """

        assert method in MOPAC_VALID_METHODS, f"MOPAC does not support {method}"

        self.cmd = cmd
        self.scr = scr
        self.method = method

        # Ensure scrdir
        # if None, use tmpdir?
        Path(scr).mkdir(parents=True, exist_ok=True)

        # Constants
        self.atomline = MOPAC_ATOMLINE
        self.filename = filename

        return


    def set_method(self, method):
        self.method = method
        return


    def get_method(self):
        return self.method


    def set_solvent(self):

        return


    def get_solvent(self):

        return


    def optimize(self,
        molobj,
        return_copy=False,
        return_properties=False,
        embed_properties=True):
        """
        TODO DOCSTRING
        """

        header = """{method} MULLIK PRECISE charge={charge} \nTITLE {title}\n"""

        if return_copy:
            molobj = copy.deepcopy(molobj)

        result_properties = self.calculate(molobj, header)

        if return_properties:
            return list(result_properties)

        for i, properties in enumerate(result_properties):

            if "coord" not in properties:
                pass
                # TODO What need to happen here? @anders

            coord = properties["coord"]

            # Set coord on conformer
            molobj_set_coordinates(molobj, coord, idx=i)

        return molobj


    def calculate(self, molobj, header):

        input_string = self._get_input_str(molobj, header, opt_flag=True)

        filename = os.path.join(self.scr, self.filename)

        with open(filename, 'w') as f:
            f.write(input_string)

        # Run mopac
        self._run_file()

        calculations = self._read_file()

        for output_lines in calculations:
            properties = get_properties(output_lines)
            yield properties

        return


    def _get_input_str(self, molobj, header, title="", opt_flag=False):
        """

        Create MOPAC input string from molobj

        """

        n_confs = molobj.GetNumConformers()

        atoms, _, charge = chembridge.molobj_to_axyzc(molobj, atom_type="str")


        txt = []

        for i in range(n_confs):

            coord = chembridge.molobj_to_coordinates(molobj, idx=i)
            header_prime = header.format(charge=charge, method=self.method, title=f"{title}_Conf_{i}")
            tx = get_input(atoms, coord, header_prime, opt_flag=opt_flag)
            txt.append(tx)

        txt = "".join(txt)

        return txt


    def _run_file(self):

        runcmd = f"{self.cmd} {self.filename}"

        stdout, stderr = shell.execute(runcmd, chdir=self.scr)

        # TODO Check stderr and stdout

        return


    def _read_file(self):

        filename = os.path.join(self.scr, self.filename)
        filename = filename.replace(".mop", ".out")

        with open(filename, 'r') as f:
            lines = f.readlines()

        molecule_lines = []

        for line in lines:

            molecule_lines.append(line.strip("\n"))

            if "TOTAL JOB TIME" in line:
                yield molecule_lines
                return

            if "CALCULATION RESULTS" in line and len(molecule_lines) > 20:
                yield molecule_lines
                molecule_lines = []

        return


    def __repr__(self):
        this = f"MopacCalc(method={self.method}, scr={self.scr}, cmd={self.cmd})"
        return this


def run_mopac(filename, cmd="", path="", hide_print=True):
    """
    """

    command = MOPCMD.format(filename)

    if hide_print:
        command += " 2> /dev/null"

    errorcode = subprocess.call(command, shell=True)

    return errorcode


def get_input(
    atoms,
    coords,
    header,
    opt_flag=False):
    """
    """

    if opt_flag:
        opt_flag=1
    else:
        opt_flag=0

    txt = header
    txt += "\n"

    for atom, coord in zip(atoms, coords):
        line = MOPAC_ATOMLINE.format(atom=atom, x=coord[0], y=coord[1], z=coord[2], opt_flag=opt_flag)
        txt += line + "\n"

    txt += "\n"

    return txt


def get_properties(lines):
    """
    TODO AUTO SWITCH

    """

    d = get_properties_optimize(lines)

    return d


def get_properties_optimize(lines):
    """
    """

    properties = {}

    # Enthalpy of formation
    idx_hof = linesio.get_rev_index(lines, "FINAL HEAT OF FORMATION")
    line = lines[idx_hof]
    line = line.split("FORMATION =")
    line = line[1]
    line = line.split()
    value = line[0]
    value = float(value)
    properties["h"] = value # kcal/mol

    # optimized coordinates
    i = linesio.get_rev_index(lines, 'CARTESIAN')
    idx_atm = 1
    idx_x = 2
    idx_y = 3
    idx_z = 4
    n_skip = 2

    j = i + n_skip
    symbols = []
    coord = []

    # continue until we hit a blank line
    while not lines[j].isspace() and lines[j].strip():
        l = lines[j].split()

        atm = l[idx_atm]
        symbols.append(atm)

        x = l[idx_x]
        y = l[idx_y]
        z = l[idx_z]
        xyz = [x, y, z]
        xyz = [float(c) for c in xyz]
        coord.append(xyz)
        j += 1

    coord = np.array(coord)
    properties["coord"] = coord
    properties["atoms"] = symbols

    return properties

