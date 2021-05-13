import copy
import os

import numpy as np

from ppqm import chembridge, constants, linesio, shell
from ppqm.calculator import BaseCalculator

MNDO_CMD = "mndo"
MNDO_ATOMLINE = "{atom:2s} {x} {opt_flag} {y} {opt_flag} {z} {opt_flag}"


class MndoCalculator(BaseCalculator):
    def __init__(self, cmd=MNDO_CMD, scr=constants.SCR, method="PM3"):

        super().__init__(scr=scr)

        self.cmd = cmd
        self.method = method

        # TODO should be a parameter
        self.read_params = False

        # Constants
        self.atomline = MNDO_ATOMLINE
        self.filename = "_tmp_mndo.inp"

    def optimize(
        self,
        molobj,
        return_copy=True,
        return_properties=False,
        read_params=False,
    ):

        header = (
            "{self.method} MULLIK PRECISE charge={charge} jprint=5\n" "nextmol=-1\nTITLE {title}"
        )

        if return_copy:
            molobj = copy.deepcopy(molobj)

        result_properties = self.calculate(molobj, header, optimize=True)

        for i, properties in enumerate(result_properties):

            if "coord" not in properties:
                pass
                # TODO What need to happen here? @anders

            properties["coord"]

            # TODO Set coord on conformer

        return molobj

    def optimize_axyzc(self, atoms, coord, charge, title=""):
        """"""

        header = (
            "{self.method} MULLIK PRECISE charge={charge} " "jprint=5\nnextmol=-1\nTITLE {title}"
        )

        properties_ = self.calculate_axyzc(atoms, coord, header, optimize=True)

        return properties_

    def calculate(self, molobj, header, optimize=False):

        input_string = self._get_input_from_molobj(
            molobj,
            self.method,
            read_params=self.read_params,
            optimize=optimize,
        )

        filename = os.path.join(self.scr, self.filename)

        with open(filename, "w") as f:
            f.write(input_string)

        calculations = self._run_mndo_file()

        for output_lines in calculations:
            properties = get_properties(output_lines)
            yield properties

        return

    def calculate_axyzc(self, atoms, coords, header, optimize=False):

        input_txt = get_input(atoms, coords, header, optimize=optimize)

        filename = os.path.join(self.scr, self.filename)

        with open(filename, "w") as f:
            f.write(input_txt)

        calculations = self._run_mndo_file()

        for output_lines in calculations:
            properties = get_properties(output_lines)
            yield properties

        return

    def _run_mndo_file(self):

        runcmd = f"{self.cmd} < {self.filename}"

        lines = shell.stream(runcmd, cwd=self.scr)

        molecule_lines = []

        for line in lines:

            molecule_lines.append(line.strip("\n"))

            if "STATISTICS FOR RUNS WITH MANY MOLECULES" in line:
                return

            if "COMPUTATION TIME" in line:
                yield molecule_lines
                molecule_lines = []

        return

    def _get_input_from_molobj(self, molobj, header, read_params=False, optimize=False, title=""):
        """"""

        # TODO Switch from header to options

        atoms, _, charge = chembridge.molobj_to_axyzc(molobj, atom_type="str")

        n_confs = molobj.GetNumConformers()

        # Create input
        txt = []
        for i in range(n_confs):
            coord = chembridge.molobj_to_coordinates(molobj, idx=i)
            # header_prime = header.format(
            #     charge=charge, method=self.method, title=f"{title}_Conf_{i}"
            # )
            tx = get_input(
                atoms,
                coord,
                header,
                read_params=self.read_params,
                optimize=optimize,
            )
            txt.append(tx)

        txt = "".join(txt)

        return txt

    def _set_input_file(self, input_str):

        # TODO Set in scr

        return

    def __repr__(self):
        return "MndoCalc(cmd={self.cmd},scr={self.scr}method={self.method})"


def get_input(atoms, coords, header, read_params=False, optimize=False):
    """
    # note: internal coordinates are assumed for three-atom systems


    """

    n_atoms = len(atoms)

    txt = header

    if read_params:
        txt = txt.split("\n")
        txt[0] += " iparok=1"
        txt = "\n".join(txt)

    txt += "\n"

    if n_atoms <= 3:
        txt += get_internal_coordinates(atoms, coords, optimize=optimize)
        txt += "\n"
        return txt

    opt_flag = 0
    if optimize:
        opt_flag = 1

    for atom, coord in zip(atoms, coords):
        fmt = {
            "atom": atom,
            "x": coord[0],
            "y": coord[1],
            "z": coord[2],
            "opt_flag": opt_flag,
        }
        line = MNDO_ATOMLINE.format(**fmt)
        txt += line + "\n"

    txt += "\n"

    return


def get_internal_coordinates(atoms, coord, optimize=False):
    """

    :param atoms: List[Str]
    :param coord: Array[]
    :param optimize: Boolean
    :return Str:
    """

    n_atoms = len(atoms)

    opt_flag = 0
    if optimize:
        opt_flag = 1

    output = ""

    if n_atoms == 3:

        ba = coord[1] - coord[0]
        bc = coord[1] - coord[2]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle) / np.pi * 180.0

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        output += f"{atoms[0]}\n"
        output += f"{atoms[1]} {norm_ba} {opt_flag}\n"
        output += f"{atoms[2]} {norm_bc} {opt_flag} {angle} {opt_flag}\n"

    elif n_atoms == 2:

        ba = coord[1] - coord[0]
        norm_ba = np.linalg.norm(ba)
        output += f"{atoms[0]}\n"
        output += f"{atoms[1]} {norm_ba} {opt_flag}\n"

    elif n_atoms == 1:

        output += f"{atoms[0]}\n"

    return output


def run_mndo_file(filename, scr=None, mndo_cmd=MNDO_CMD):

    # TODO Needed here? or force people to use class

    return


def get_properties(output):
    """"""

    if isinstance(output, str):
        output = output.split("\n")

    result = get_properties_optimize(output)

    # TODO Read keywords to detect property type

    # TODO Detect failures

    return result


def get_properties_1scf(lines):

    properties = {}

    # Check if input coordiantes is internal
    # INPUT IN INTERNAL COORDINATES
    # INPUT IN CARTESIAN COORDINATES
    idx = linesio.get_index(lines, "INPUT IN")
    line = lines[idx]
    is_internal = "INTERNAL" in line

    keywords = [
        "CORE HAMILTONIAN MATRIX.",
        "NUCLEAR ENERGY",
        "IONIZATION ENERGY",
        "INPUT GEOMETRY",
    ]

    idx_keywords = linesio.get_rev_indexes(lines, keywords)

    # SCF energy
    idx_core = idx_keywords[0]
    if idx_core is None:
        e_scf = float("nan")
        properties["e_scf"] = e_scf
    else:
        idx = idx_core
        idx -= 9
        line = lines[idx]

        if "SCF CONVERGENCE HAS BEE" in line:
            idx -= 2
            line = lines[idx]

        # NOTE This should never happen, but better safe than sorry
        line = line.split()
        if len(line) < 2:
            e_scf = float("nan")
        else:
            value = line[1]
            e_scf = float(value)

        properties["e_scf"] = e_scf  # ev

    # Nuclear energy
    if idx_keywords[1] is None:
        e_nuc = float("nan")
        properties["e_nuc"] = e_nuc
    else:
        idx = idx_keywords[1]
        line = lines[idx]
        line = line.split()
        value = line[2]
        e_nuc = float(value)
        properties["e_nuc"] = e_nuc  # ev

    # eisol
    eisol = dict()
    idxs = linesio.get_indexes_with_stop(lines, "EISOL", "IDENTIFICATION")
    for idx in idxs:
        line = lines[idx]
        line = line.split()
        atom = int(line[0])
        value = line[2]
        eisol[atom] = float(value)  # ev

    # # Enthalpy of formation
    idx_hof = linesio.get_index(lines, "SCF HEAT OF FORMATION")
    line = lines[idx_hof]
    line = line.split("FORMATION")
    line = line[1]
    line = line.split()
    value = line[0]
    value = float(value)
    properties["h"] = value  # kcal/mol

    # ionization
    # idx = get_rev_index(lines, "IONIZATION ENERGY")
    idx = idx_keywords[2]
    if idx is None:
        e_ion = float("nan")
        properties["e_ion"] = e_ion
    else:
        line = lines[idx]
        value = line.split()[-2]
        e_ion = float(value)  # ev
        properties["e_ion"] = e_ion

    # # Dipole
    # idx = get_rev_index(lines, "PRINCIPAL AXIS")
    # line = lines[idx]
    # line = line.split()
    # value = line[-1]
    # value = float(value) # Debye
    # properties["mu"] = value

    # input coords

    atoms = []
    coord = []

    if is_internal:

        idx_atm = 1
        idx_x = 2
        idx_y = 3
        idx_z = 4

        idx_coord = linesio.get_index(lines, "INITIAL CARTESIAN COORDINATES")
        idx_coord += 5

        j = idx_coord
        # continue until we hit a blank line
        while not lines[j].isspace() and lines[j].strip():
            line = lines[j].split()

            atom = line[idx_atm]
            atom = int(atom)
            x = float(line[idx_x])
            y = float(line[idx_y])
            z = float(line[idx_z])

            atoms.append(atom)
            coord.append([x, y, z])

            j += 1

    else:

        idx_atm = 1
        idx_x = 2
        idx_y = 3
        idx_z = 4

        idx = idx_keywords[3]
        idx += 6

        j = idx
        # continue until we hit a blank line
        while not lines[j].isspace() and lines[j].strip():
            line = lines[j].split()
            atoms.append(int(line[idx_atm]))
            x = line[idx_x]
            y = line[idx_y]
            z = line[idx_z]
            xyz = [x, y, z]
            xyz = [float(c) for c in xyz]
            coord.append(xyz)
            j += 1

    # calculate energy
    e_iso = [eisol[a] for a in atoms]
    e_iso = np.sum(e_iso)
    energy = e_nuc + e_scf - e_iso

    properties["energy"] = energy

    return properties


def get_properties_optimize(lines):
    """

    TODO Read how many steps

    """

    properties = {}

    # # Enthalpy of formation
    idx_hof = linesio.get_index(lines, "SCF HEAT OF FORMATION")
    line = lines[idx_hof]
    line = line.split("FORMATION")
    line = line[1]
    line = line.split()
    value = line[0]
    value = float(value)
    properties["h"] = value  # kcal/mol

    # optimized coordinates
    i = linesio.get_rev_index(lines, "CARTESIAN COORDINATES")
    idx_atm = 1
    idx_x = 2
    idx_y = 3
    idx_z = 4
    n_skip = 4

    if i < idx_hof:
        i = linesio.get_rev_index(lines, "X-COORDINATE")
        idx_atm = 1
        idx_x = 2
        idx_y = 4
        idx_z = 6
        n_skip = 3

    j = i + n_skip
    symbols = []
    coord = []

    # continue until we hit a blank line
    while not lines[j].isspace() and lines[j].strip():
        line = lines[j].split()
        symbols.append(int(line[idx_atm]))
        x = line[idx_x]
        y = line[idx_y]
        z = line[idx_z]
        xyz = [x, y, z]
        xyz = [float(c) for c in xyz]
        coord.append(xyz)
        j += 1

    coord = np.array(coord)
    properties["coord"] = coord
    properties["atoms"] = symbols

    return


def get_properties_gradient():
    """"""

    return
