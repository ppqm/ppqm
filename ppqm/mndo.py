from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np

from ppqm import chembridge, constants
from ppqm.calculator import BaseCalculator
from ppqm.chembridge import Mol
from ppqm.utils import linesio, shell

MNDO_CMD = "mndo"
MNDO_ATOMLINE = "{atom:2s} {x} {opt_flag} {y} {opt_flag} {z} {opt_flag}"


class MndoCalculator(BaseCalculator):
    def __init__(
        self,
        cmd: str = MNDO_CMD,
        scr: Path = constants.SCR,
        n_cores: int = 1,
        show_progress: bool = False,
    ) -> None:

        super().__init__(scr=scr)

        self.cmd = cmd

        # TODO should be a parameter
        self.read_params = False

        # Constants
        self.atomline = MNDO_ATOMLINE
        self.default_filename = "_tmp_mndo.inp"

        self.n_cores = n_cores
        self.show_progress = show_progress

        # "{self.method} MULLIK PRECISE charge={charge} " "jprint=5\nnextmol=-1\nTITLE {title}"
        self.default_options: Dict[str, Any] = {
            "mullik": None,
            "precise": None,
            "jprint": 5,
        }

    # def optimize(
    #     self,
    #     molobj: Mol,
    #     return_copy: bool = True,
    #     read_params: bool = False,
    # )->Mol:
    #
    #     raise NotImplementedError
    #
    #     header = (
    #         "{self.method} MULLIK PRECISE charge={charge} jprint=5\n" "nextmol=-1\nTITLE {title}"
    #     )
    #
    #     if return_copy:
    #         molobj = copy.deepcopy(molobj)
    #
    #     result_properties = self.calculate(molobj, header, optimize=True)
    #
    #     for _, properties in enumerate(result_properties):
    #
    #         if "coord" not in properties:
    #             pass
    #             # TODO What need to happen here? @anders
    #
    #         properties["coord"]
    #
    #         # TODO Set coord on conformer
    #
    #     return molobj

    # def optimize_axyzc(self, atoms, coord, charge, title=""):
    #     """"""
    #     raise NotImplementedError
    #
    #     header = (
    #         "{self.method} MULLIK PRECISE charge={charge} " "jprint=5\nnextmol=-1\nTITLE {title}"
    #     )
    #
    #     properties_ = self.calculate_axyzc(atoms, coord, header, optimize=True)
    #
    #     return properties_

    def calculate(
        self, molobj: Mol, options: dict, optimize: bool = False
    ) -> List[Optional[dict]]:

        input_string = self._get_input_from_molobj(
            molobj,
            options,
            read_params=self.read_params,
            optimize=optimize,
        )

        filename = self.scr / self.default_filename

        # TODO Split into multiple files, based on cores
        with open(filename, "w") as f:
            f.write(input_string)

        calculations = self._run_mndo_file(filename, scr=self.scr)

        result: List[Optional[dict]] = [get_properties(lines) for lines in calculations]

        return result

    def _run_mndo_file(
        self, filename: Path, scr: Optional[Path] = None
    ) -> Generator[List[str], None, None]:

        runcmd = f"{self.cmd} < {filename}"

        lines = shell.stream(runcmd, cwd=scr)

        molecule_lines = []

        for line in lines:

            molecule_lines.append(line.strip("\n"))

            if "STATISTICS FOR RUNS WITH MANY MOLECULES" in line:
                return

            if "COMPUTATION TIME" in line:
                yield molecule_lines
                molecule_lines = []

        return

    def _get_input_from_molobj(
        self, molobj: Mol, options: dict, read_params: bool = False, optimize: bool = False
    ) -> str:
        """"""

        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

        options = {**options, "charge": charge}
        header = get_header(options)

        n_confs = molobj.GetNumConformers()

        # Create input
        txt = []
        for i in range(n_confs):
            coord = chembridge.get_coordinates(molobj, confid=i)
            # header_prime = header.format(
            #     charge=charge, method=self.method, title=f"{title}_Conf_{i}"
            # )
            tx = get_input(
                atoms,
                coord,
                header,
                read_params=read_params,
                optimize=optimize,
            )
            txt.append(tx)

        return "".join(txt)

    def __repr__(self) -> str:
        return f"MndoCalc(cmd={self.cmd},scr={self.scr},n_cores={self.n_cores})"


def get_header(options: dict) -> str:
    """return mndoheader from options dict"""

    "{self.method} MULLIK PRECISE charge={charge} jprint=5\nnextmol=-1\nTITLE {title}"

    title = options.get("title", "TITLE")
    if "title" in options:
        del options["title"]

    header: List[Any] = [""] * 4
    header[0] = list()
    header[1] = "nnextmol=-1"
    header[2] = title

    for key, val in options.items():

        if val is not None:
            keyword = f"{key}={val}"
        else:
            keyword = f"{key}"

        header[0].append(keyword)

    header[0] = " ".join(header[0])
    return "\n".join(header)


def get_input(
    atoms: Union[List[str], np.ndarray],
    coords: np.ndarray,
    header: str,
    read_params: bool = False,
    optimize: bool = False,
) -> str:
    """
    # note: internal coordinates are assumed for three-atom systems


    """

    n_atoms = len(atoms)

    txt = header

    if read_params:
        txt_ = txt.split("\n")
        txt_[0] += " iparok=1"
        txt = "\n".join(txt_)

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

    return txt


def get_internal_coordinates(
    atoms: Union[List[str], np.ndarray], coord: np.ndarray, optimize: bool = False
) -> str:
    """Get MNDO input in internal coordinates format"""

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


def get_properties(output: List[str]) -> Optional[dict]:
    """"""

    if isinstance(output, str):
        output = output.split("\n")

    result = get_properties_optimize(output)

    # TODO Read keywords to detect property type

    # TODO Detect failures

    return result


def get_properties_1scf(lines: List[str]) -> Optional[dict]:

    properties = {}

    # Check if input coordiantes is internal
    # INPUT IN INTERNAL COORDINATES
    # INPUT IN CARTESIAN COORDINATES
    idx = linesio.get_index(lines, "INPUT IN")
    assert idx is not None, "Uncaught MNDO error"
    line = lines[idx]
    is_internal = "INTERNAL" in line

    keywords = [
        "CORE HAMILTONIAN MATRIX.",
        "NUCLEAR ENERGY",
        "IONIZATION ENERGY",
        "INPUT GEOMETRY",
    ]

    idx_keywords = linesio.get_rev_indices_patterns(lines, keywords)

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
        line_ = line.split()
        if len(line_) < 2:
            e_scf = float("nan")
        else:
            value = line_[1]
            e_scf = float(value)

        properties["e_scf"] = e_scf  # ev

    # Nuclear energy
    if idx_keywords[1] is None:
        e_nuc = float("nan")
        properties["e_nuc"] = e_nuc
    else:
        idx = idx_keywords[1]
        line = lines[idx]
        line_ = line.split()
        value = line_[2]
        e_nuc = float(value)
        properties["e_nuc"] = e_nuc  # ev

    # eisol
    eisol = dict()
    idxs = linesio.get_rev_indices(lines, "EISOL", stoppattern="IDENTIFICATION")
    for idx in idxs:
        line = lines[idx]
        line_ = line.split()
        atom = int(line_[0])
        value = line[2]
        eisol[atom] = float(value)  # ev

    # # Enthalpy of formation
    idx_hof = linesio.get_index(lines, "SCF HEAT OF FORMATION")
    assert idx_hof is not None, "Uncaught MNDO exception"
    line = lines[idx_hof]
    line_ = line.split("FORMATION")
    line = line_[1]
    line_ = line.split()
    value = line_[0]
    properties["h"] = float(value)  # kcal/mol

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
        assert idx_coord is not None, "Uncaught MNDO error"
        idx_coord += 5

        j = idx_coord
        # continue until we hit a blank line
        while not lines[j].isspace() and lines[j].strip():
            line_ = lines[j].split()

            atom = int(line_[idx_atm])
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
        assert idx is not None, "Uncaght MNDO error"
        idx += 6

        j = idx
        # continue until we hit a blank line
        while not lines[j].isspace() and lines[j].strip():
            line_ = lines[j].split()
            atoms.append(int(line_[idx_atm]))
            x = float(line_[idx_x])
            y = float(line_[idx_y])
            z = float(line_[idx_z])
            xyz = [x, y, z]
            coord.append(xyz)
            j += 1

    # calculate energy
    e_iso_ = [eisol[a] for a in atoms]
    e_iso = np.sum(e_iso_)
    energy = e_nuc + e_scf - e_iso

    properties["energy"] = energy

    return properties


def get_properties_optimize(lines: List[str]) -> Optional[dict]:
    """

    TODO Read how many steps

    """

    properties: Dict[str, Any] = {}

    # # Enthalpy of formation
    idx_hof = linesio.get_index(lines, "SCF HEAT OF FORMATION")
    assert idx_hof is not None, "Uncaught MNDO error"
    line = lines[idx_hof]
    line_ = line.split("FORMATION")
    line = line_[1]
    line_ = line.split()
    value = line_[0]
    properties["h"] = float(value)  # kcal/mol

    # optimized coordinates
    i = linesio.get_rev_index(lines, "CARTESIAN COORDINATES")
    assert i is not None, "Uncaught MNDO error"
    idx_atm = 1
    idx_x = 2
    idx_y = 3
    idx_z = 4
    n_skip = 4

    if i < idx_hof:
        i = linesio.get_rev_index(lines, "X-COORDINATE")
        assert i is not None, "Uncaught MNDO error"
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
        line_ = lines[j].split()
        symbols.append(int(line_[idx_atm]))
        x = float(line[idx_x])
        y = float(line[idx_y])
        z = float(line[idx_z])
        xyz = [x, y, z]
        coord.append(xyz)
        j += 1

    properties["coord"] = np.array(coord)
    properties["atoms"] = symbols

    return properties
