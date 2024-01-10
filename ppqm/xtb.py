"""
xTB wrapper functions
"""

import functools
import logging
import multiprocessing
import os
from collections import ChainMap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rmsd  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from ppqm import WorkDir, chembridge, constants
from ppqm.calculator import BaseCalculator
from ppqm.chembridge import Mol
from ppqm.utils import func_parallel, linesio, shell

# from rdkit.Chem.rdchem import Mol   # type: ignore[import-untyped]


XTB_CMD = "xtb"
XTB_FILENAME = "_tmp_xtb_input.xyz"
XTB_FILES = ["charges", "wbo", "xtbrestart", "xtbopt.coord", "xtbopt.log"]

COLUMN_ENERGY = "total_energy"
COLUMN_COORD = "coord"
COLUMN_ATOMS = "atoms"
COLUMN_GSOLV = "gsolv"
COLUMN_DIPOLE = "dipole"
COLUMN_CONVERGED = "is_converged"
COLUMN_STEPS = "opt_steps"

_logger = logging.getLogger(__name__)


class XtbCalculator(BaseCalculator):
    """Wrapper for xTB. Please read more on xtb-docs.readthedocs.io"""

    def __init__(
        self,
        cmd: str = XTB_CMD,
        filename: str = XTB_FILENAME,
        show_progress: bool = False,
        n_cores: int = 1,
        keep_files: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.cmd: str = cmd
        self.filename: str = filename
        self.n_cores: int = n_cores
        self.show_progress: bool = show_progress

        if n_cores < 0:
            n_cores = multiprocessing.cpu_count()

        # TODO Should not be using xtb_options
        self.xtb_options: Dict[str, Any] = dict(
            cmd=self.cmd,
            scr=self.scr,
            filename=self.filename,
            keep_files=keep_files,
        )

        # Default xtb options
        self.options: Dict = {}

        # Check version and command
        self.health_check()

    def health_check(self) -> None:

        assert shell.which(self.cmd), f"Cannot find {self.cmd}"

        stdout, _ = shell.execute(f"{self.cmd} --version")

        assert stdout is not None

        try:
            stdout_lines = stdout.split("\n")
            stdout_lines = [x for x in stdout_lines if "*" in x]
            version = stdout_lines[0].strip().split()
            version = version[3].split(".")
            major, minor, _ = version
        except Exception:
            assert False, "too old xtb version"

        assert int(major) >= 6, "too old xtb version"
        assert int(minor) >= 4, "too old xtb version"

    def calculate(self, molobj: Mol, options: dict, **kwargs: Any) -> List[Optional[dict]]:

        # Merge options
        options_prime = dict(ChainMap(options, self.options))

        if self.n_cores and self.n_cores > 1:
            return self.calculate_parallel(molobj, options_prime, **kwargs)

        return self.calculate_serial(molobj, options_prime, **kwargs)

    def calculate_serial(self, molobj: Mol, options: dict, **kwargs: Any) -> List[Optional[dict]]:

        properties_list = []
        n_confs = molobj.GetNumConformers()

        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

        pbar = None
        if self.show_progress:
            pbar = tqdm(
                total=n_confs,
                desc="xtb(1)",
                **constants.TQDM_OPTIONS,
            )

        for conf_idx in range(n_confs):

            coord = chembridge.get_coordinates(molobj, confid=conf_idx)

            properties = get_properties_from_axyzc(
                atoms, coord, charge, options, **self.xtb_options, **kwargs
            )

            properties_list.append(properties)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return properties_list

    def calculate_parallel(
        self, molobj: Mol, options: dict, n_cores: int = 1
    ) -> List[Optional[dict]]:

        _logger.debug("start xtb multiprocessing pool")

        if not n_cores:
            n_cores = self.n_cores

        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)
        n_conformers: int = molobj.GetNumConformers()

        coordinates_list = [
            np.asarray(conformer.GetPositions()) for conformer in molobj.GetConformers()  # type: ignore[attr-defined]
        ]

        n_procs: int = min(n_cores, n_conformers)
        results = []

        func = functools.partial(
            get_properties_from_acxyz, atoms, charge, options=options, **self.xtb_options
        )

        results = func_parallel(
            func,
            coordinates_list,
            n_cores=n_procs,
            n_jobs=n_conformers,
            show_progress=self.show_progress,
            title="XTB",
        )

        return results

    def __repr__(self) -> str:
        return f"XtbCalc(cmd={self.cmd},scr={self.scr},n_cores={self.n_cores})"


def get_properties_from_acxyz(
    atoms: Union[List[str], np.ndarray], charge: int, coordinates: np.ndarray, **kwargs: Any
) -> Optional[dict]:
    """get properties from atoms, charge and coordinates"""
    return get_properties_from_axyzc(atoms, coordinates, charge, **kwargs)


def get_properties_from_axyzc(
    atoms_str: Union[List[str], np.ndarray],
    coordinates: np.ndarray,
    charge: int,
    options: Optional[dict] = None,
    scr: Path = constants.SCR,
    keep_files: bool = False,
    cmd: str = XTB_CMD,
    filename: str = XTB_FILENAME,
    n_cores: int = 1,
) -> Optional[dict]:
    """Get XTB properties from atoms, coordinates and charge for a molecule."""

    if not filename.endswith(".xyz"):
        filename += ".xyz"

    workdir = WorkDir(dir=scr, prefix="xtb_", keep=keep_files)
    temp_scr = workdir.get_path()

    _logger.debug(f"xtb work dir {temp_scr}")

    # Write input file (XYZ format)
    inputstr = rmsd.set_coordinates(atoms_str, coordinates, title="xtb input")

    with open(temp_scr / filename, "w") as f:
        f.write(inputstr)

    # Set charge in file
    with open(temp_scr / ".CHRG", "w") as f:
        f.write(str(charge))

    # Overwrite threads
    shell.set_threads(n_cores)

    # Run subprocess command
    cmd_ = [cmd, f"{filename}"]

    if options:
        cmd_ += parse_options(options)

    # Merge to string
    cmd_full = " ".join(cmd_)

    _logger.debug(cmd_full)

    stdout, stderr = shell.execute(cmd_full, cwd=temp_scr)
    assert stdout is not None
    assert stderr is not None
    lines = stdout.split("\n") + stderr.split("\n")

    error_pattern = "abnormal termination of xtb"
    idx = linesio.get_rev_index(lines, error_pattern, stoppattern="#")
    if idx is not None:

        _logger.critical(error_pattern)

        idx = linesio.get_rev_index(lines, "ERROR")

        if idx is None:
            _logger.critical("could not read error message")

        else:

            for line in lines[idx + 1 : -2]:
                _logger.critical(line.strip())

        _logger.critical(cmd_full)
        _logger.critical("xtbexec " + str(shell.which(cmd)))
        _logger.critical("xtbpath " + os.environ.get("XTBPATH", ""))
        _logger.critical("xtbhome " + os.environ.get("XTBHOME", ""))

        return None

    # Parse properties from xtb output
    properties = read_properties(lines, options=options, scr=temp_scr)

    return properties


# Readers


def read_status(lines: List[str]) -> bool:
    """Did xtb end normally?"""
    keywords = [
        "Program stopped due to fatal error",
        "abnormal termination of xtb",
    ]
    stoppattern = "normal termination of xtb"

    idxs = linesio.get_rev_indices_patterns(lines, keywords, stoppattern=stoppattern)

    for idx in idxs:
        if idx is not None:
            return False

    return True


def parse_sum_table(lines: List[str]) -> dict:
    """Parse the summary table from xtb log"""

    properties = dict()

    for line in lines:

        if ":::" in line:
            continue

        if "..." in line:
            continue

        # Needs a loop break when the Hessian is computed.
        if "Hessian" in line:
            break

        line_ = (
            line.replace("w/o", "without")
            .replace(":", "")
            .replace("->", "")
            .replace("/", "_")
            .strip()
        ).split()

        if len(line_) < 2:
            continue

        value = float(line_[-2])
        # unit = line[-1]
        name_ = line_[:-2]
        name = "_".join(name_).lower()
        name = name.replace("-", "_").replace(".", "")

        properties[name] = float(value)

    return properties


def read_properties(
    lines: List[str], options: Optional[dict] = None, scr: Optional[Path] = None
) -> Optional[dict]:
    """Read output based on options or output"""

    reader: Optional[Callable] = None
    properties: Optional[dict]
    read_files = True

    if options is None:
        reader = read_properties_opt

    elif "vfukui" in options:
        reader = read_properties_fukui
        read_files = False

    elif "vomega" in options:
        reader = read_properties_omega
        read_files = False

    elif "opt" in options or "ohess" in options:
        reader = read_properties_opt

    if reader is None:
        reader = read_properties_sp

    properties = reader(lines)

    if properties is None:
        return None

    assert isinstance(properties, dict)

    if scr is not None and read_files:
        # Parse file properties

        files = [path.name for path in scr.iterdir()]

        if "charges" in files:
            charges = get_mulliken_charges(scr=scr)
            properties["mulliken_charges"] = charges

        if "wbo" in files:
            bonds, bondorders = get_wbo(scr=scr)
            properties["bonds"] = bonds
            properties["bondorders"] = bondorders

        # TODO Not sure if this should be here
        properties.update(get_cm5_charges(lines))  # Can return {} if not GFN1

        if "vibspectrum" in files:
            properties["frequencies"] = get_frequencies(scr=scr)

    return properties


def read_properties_sp(lines: List[str]) -> Optional[dict]:
    """
    TODO read dipole moment
    TODO Inlcude units in docstring
    TODO GEOMETRY OPTIMIZATION CONVERGED AFTER 48 ITERATIONS

    electornic_energy is SCC energy

    """

    # TODO Better logging for crashed xtb
    if not read_status(lines):
        return None

    keywords = [
        "final structure:",
        "::                     SUMMARY                     ::",
        "Property Printout  ",
    ]

    stoppattern = "CYCLE    "
    idxs = linesio.get_rev_indices_patterns(lines, keywords, stoppattern=stoppattern)

    assert idxs[1] is not None, "Uncaught xtb exception. Please report."
    assert idxs[2] is not None, "Uncaught xtb exception. Please report."

    idx_coord = idxs[0]
    idx_summary = idxs[1]
    idx_end_summary = idxs[2]

    # Get atom count
    # keyword = "number of atoms"
    # idx = linesio.get_index(lines, keyword)
    # assert idx is not None, "Uncaught xtb exception. Should not happen"
    # line = lines[idx]
    # n_atoms_ = line.split()[-1]
    # n_atoms = int(n_atoms_)

    # Get energies
    idx_summary = idxs[1] + 1

    # :: total energy        +1
    # :: total w/o Gsasa/hb  +2
    # :: gradient norm       +3
    # :: HOMO-LUMO gap       +4
    # ::.....................+4
    # :: SCC energy          +5
    # :: -> isotropic ES     +6
    # :: -> anisotropic ES   +7
    # :: -> anisotropic XC   +8
    # :: -> dispersion       +9
    # :: -> Gsolv            +10
    # ::    -> Gborn         +11
    # ::    -> Gsasa         +12
    # ::    -> Ghb           +13
    # ::    -> Gshift        +14
    # :: repulsion energy    +15
    # :: add. restraining    +16

    prop_lines = lines[idx_summary : idx_end_summary - 2]
    prop_dict = parse_sum_table(prop_lines)

    # total_energy = prop_dict.get("total_energy", float("nan"))
    # gsolv = prop_dict.get("gsolv", float("nan"))
    # electronic_energy = prop_dict.get("scc_energy", float("nan"))

    properties = prop_dict

    # Get dipole
    dipole_str = "molecular dipole:"
    idx = linesio.get_rev_index(lines, dipole_str)
    if idx is None:
        dipole_tot = None
    else:
        idx += 3
        line = lines[idx]
        line_ = line.split()
        dipole_tot = float(line_[-1])

    properties = {
        COLUMN_DIPOLE: dipole_tot,
        **properties,
    }

    if idx_coord is not None:
        try:
            num_atoms = int(lines[idx_coord + 2])
            properties["n_atoms"] = num_atoms
        except ValueError:
            _logger.error("Unable to parse number of atoms")

    # Get covalent properties
    properties_covalent = read_covalent_coordination(lines)
    if properties_covalent is not None:
        properties = {**properties, **properties_covalent}

    # Get orbitals
    properties_orbitals = read_properties_orbitals(lines)
    if properties_orbitals is not None:
        properties = {**properties, **properties_orbitals}

    return properties


def read_properties_opt(lines: List[str]) -> Optional[dict]:
    """

    electornic_energy is SCC energy

    """

    # TODO Better logging for crashed xtb
    if not read_status(lines):
        return None

    keywords = [
        "final structure:",
        "ITERATIONS",
    ]

    stoppattern = "CYCLE    "
    idxs = linesio.get_rev_indices_patterns(lines, keywords, stoppattern=stoppattern)
    idx_coord = idxs[0]
    idx_optimization = idxs[1]

    properties = read_properties_sp(lines)
    assert properties is not None, "Uncaught error"
    assert "n_atoms" in properties, "Unable to parse number of atoms"

    n_atoms: int = properties["n_atoms"]

    atoms: Optional[Union[List, np.ndarray]]
    coords: Optional[Union[List, np.ndarray]]

    # Get coordinates
    if idx_coord is None:
        coords = None
        atoms = None

    else:

        def parse_coordline(line: str) -> Tuple[str, List[float]]:
            line_ = line.split()
            atom = line_[0]
            coord = [float(x) for x in line_[1:]]
            return atom, coord

        atoms = []
        coords = []

        for i in range(idx_coord + 4, idx_coord + 4 + n_atoms):
            line = lines[i]
            atom, coord = parse_coordline(line)
            atoms.append(atom)
            coords.append(coord)

        atoms = np.array(atoms)
        coords = np.array(coords)

    # Get dipole
    dipole_str = "molecular dipole:"
    idx = linesio.get_rev_index(lines, dipole_str)
    if idx is None:
        dipole_tot = None
    else:
        idx += 3
        line = lines[idx]
        line_ = line.split()
        dipole_tot = float(line_[-1])

    if idx_optimization is None:
        is_converged = None
        n_cycles = None

    else:

        line = lines[idx_optimization]
        if "FAILED" in line:
            is_converged = False
        else:
            is_converged = True

        line_ = line.split()
        n_cycles = int(line_[-3])

    # Get covCN and alpha
    properties_covalent = read_covalent_coordination(lines)
    assert properties_covalent is not None

    properties = {
        COLUMN_ATOMS: atoms,
        COLUMN_COORD: coords,
        COLUMN_DIPOLE: dipole_tot,
        COLUMN_CONVERGED: is_converged,
        COLUMN_STEPS: n_cycles,
        **properties_covalent,
        **properties,
    }

    return properties


def read_properties_omega(lines: List[str]) -> Optional[dict]:
    """


    Format:

    Calculation of global electrophilicity index (IP+EA)²/(8·(IP-EA))
    Global electrophilicity index (eV):    0.0058
    """

    keywords = ["Global electrophilicity index"]
    indices = linesio.get_rev_indices_patterns(lines, keywords)

    if indices[0] is None:
        return None

    line = lines[indices[0]]
    line_ = line.split()
    global_index = float(line_[-1])

    properties = {"global_electrophilicity_index": global_index}

    return properties


def read_properties_fukui(lines: List[str]) -> Optional[dict]:
    """
    Read the Fukui properties fro XTB log

    format:
     #        f(+)     f(-)     f(0)
     1O      -0.086   -0.598   -0.342
     2H      -0.457   -0.201   -0.329
     3H      -0.457   -0.201   -0.329
    """

    keywords = ["Fukui index Calculation", "f(+)", "Property Printout"]

    indices = linesio.get_rev_indices_patterns(lines, keywords)

    if indices[0] is None or indices[1] is None or indices[2] is None:
        return None

    start_index = indices[1]
    end_index = indices[2]

    f_plus_list = list()
    f_minus_list = list()
    f_zero_list = list()

    for i in range(start_index + 1, end_index - 1):
        line = lines[i]
        line_ = line.split()

        f_plus = float(line_[1])
        f_minus = float(line_[2])
        f_zero = float(line_[3])

        f_plus_list.append(f_plus)
        f_minus_list.append(f_minus)
        f_zero_list.append(f_zero)

    f_plus_vec = np.array(f_plus_list)
    f_minus_vec = np.array(f_minus_list)
    f_zero_vec = np.array(f_zero_list)

    properties = {
        "f_plus": f_plus_vec,
        "f_minus": f_minus_vec,
        "f_zero": f_zero_vec,
    }

    return properties


def get_mulliken_charges(scr: Optional[Path] = None) -> Optional[np.ndarray]:

    if scr is None:
        scr = Path(".")

    filename = scr / "charges"

    if not filename.is_file():
        _logger.error("Unable to read 'charges' in {scr}")
        return None

    # read charges files from work dir
    charges = np.loadtxt(filename)

    return charges


def get_cm5_charges(lines: List[str]) -> dict:
    """Get CM5 charges from gfn1-xTB calculation"""

    keywords = ["Mulliken/CM5 charges", "Wiberg/Mayer (AO) data"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)

    if start is None:  # No CM5 charges -> not GFN1 calculation
        return {}

    cm5_charges = []
    for line in lines[start + 1 : stop]:
        line = line.strip()
        if line:
            cm5_charges.append(float(line.split()[2]))

    return {"cm5_charges": cm5_charges}


def get_wbo(scr: Optional[Path] = None) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Get wiberg bonds and borders from xtb result folder"""

    if scr is None:
        scr = Path(".")

    filename = scr / "wbo"

    if not filename.is_file():
        return [], []

    # Read WBO file
    with open(filename, "r") as f:
        lines = f.readlines()

    bonds, bondorders = read_wbo(lines)

    return bonds, bondorders


def read_wbo(lines: List[str]) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Read Wiberg bond order lines"""
    # keyword = "Wiberg bond orders"

    bonds = []
    bondorders = []
    for line in lines:
        parts = line.strip().split()
        bondorders.append(float(parts[-1]))
        parts = parts[:2]
        parts_ = [int(x) - 1 for x in parts]
        parts__ = (min(parts_), max(parts_))
        bonds.append(parts__)

    return bonds, bondorders


def read_properties_orbitals(lines: List[str], n_offset: int = 2) -> Optional[dict]:
    """

    format:

         #    Occupation            Energy/Eh            Energy/eV
      -------------------------------------------------------------
       ...           ...                  ...                  ...
        62        2.0000           -0.3635471              -9.8926
        63        2.0000           -0.3540913              -9.6353 (HOMO)
        64                         -0.2808508              -7.6423 (LUMO)
       ...           ...                  ...                  ...

    """

    properties = dict()

    keywords = ["(HOMO)", "(LUMO)"]
    indices = linesio.get_rev_indices_patterns(lines, keywords)

    if indices[0] is None or indices[1] is None:
        return None

    idx_homo = indices[0]
    idx_lumo = indices[1]

    # check if this is the right place
    if idx_homo - idx_lumo != -1:
        return None

    # HOMO
    line = lines[idx_homo]
    line_ = line.split()
    energy_homo = float(line_[2])

    properties["homo"] = energy_homo

    # HOMO Offsets
    for i in range(n_offset):
        line = lines[idx_homo - (i + 1)]
        line_ = line.strip().split()

        if len(line_) < 3:
            continue

        value = line_[2]
        properties[f"homo-{i+1}"] = float(value)

    # LUMO
    line = lines[idx_lumo]
    line_ = line.split()
    idx_lumo_col = 1
    energy_lumo = float(line_[idx_lumo_col])

    properties["lumo"] = energy_lumo

    # Lumo Offsets
    for i in range(n_offset):
        line = lines[idx_lumo + (i + 1)]
        line_ = line.strip().split()

        if len(line_) < 3:
            continue

        value = line_[idx_lumo_col]
        properties[f"lumo+{i+1}"] = float(value)

    return properties


def read_covalent_coordination(lines: List[str]) -> Optional[dict]:
    """
    Read computed covalent coordination number.

    format:

    #   Z          covCN         q      C6AA      α(0)
    1   6 C        3.743    -0.105    22.589     6.780
    2   6 C        3.731     0.015    20.411     6.449
    3   7 N        2.732    -0.087    22.929     7.112
    ...

    Mol. C6AA /au·bohr

    """
    properties: dict = {"covCN": [], "alpha": []}

    start_line = linesio.get_rev_index(lines, "covCN")

    if start_line is None:
        return None

    for line in lines[start_line + 1 :]:
        if set(line).issubset(set(["\n"])):
            break

        line_ = line.strip().split()
        covCN = float(line_[3])
        alpha = float(line_[-1])

        properties["covCN"].append(covCN)
        properties["alpha"].append(alpha)

    return properties


def get_frequencies(scr: Optional[Path] = None) -> List[float]:
    """ """

    if scr is None:
        scr = Path(".")

    filename = scr / "vibspectrum"

    # Read WBO file
    with open(filename, "r") as f:
        lines = f.readlines()

    frequencies = read_frequencies(lines)
    return frequencies


def read_frequencies(lines: List[str]) -> List[float]:
    """ """
    frequencies = []
    for line in lines[3:]:

        if "$end" in line:
            break
        if "-" in line:  # non vib modes
            continue
        frequencies.append(float(line.strip().split()[2]))

    return frequencies


def parse_options(options: dict) -> List[str]:
    """Parse dictionary/json of options, and return arg list for xtb"""

    cmd_options: List[str] = []

    for key, value in options.items():

        txt: str
        if value is not None:
            txt = f"--{key} {value}"
        else:
            txt = f"--{key}"

        cmd_options.append(txt)

    return cmd_options
