import logging
import os
import tempfile
from collections import ChainMap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ppqm import chembridge, constants
from ppqm.calculator import BaseCalculator
from ppqm.chembridge import Mol
from ppqm.utils import linesio, shell

GAMESS_CMD = "rungms"
GAMESS_SCR = Path("~/scr/")
GAMESS_USERSCR = Path("~/scr/")
GAMESS_ATOMLINE = "{:2s}    {:2.1f}    {:f}     {:f}    {:f}"
GAMESS_FILENAME = "_tmp_gamess.inp"
GAMESS_SQM_METHODS = ["AM1", "PM3", "PM6"]
GAMESS_KEYWORD_CHARGE = "{charge}"

GAMESS_DEFAULT_OPTIONS = {"method": "pm3"}

COLUMN_THERMO = "thermo"
COLUMN_CHARGES = "charges"
COLUMN_SOLV_TOTAL = "solvation_total"
COLUMN_SOLV_POLAR = "solvation_polar"
COLUMN_SOLV_NONPOLAR = "solvation_nonpolar"
COLUMN_SOLV_SURFACE = "surface"
COLUMN_TOTAL_CHARGE = "total_charge"
COLUMN_DIPOLE_VEC = "dipole"
COLUMN_DIPOLE_TOTAL = "dipole_total"

_logger = logging.getLogger(__name__)

random_names = tempfile._get_candidate_names()  # type: ignore[attr-defined]


class GamessCalculator(BaseCalculator):
    def __init__(
        self,
        cmd: str = GAMESS_CMD,
        filename: Optional[str] = None,
        gamess_scr: Path = GAMESS_SCR,
        gamess_userscr: Path = GAMESS_USERSCR,
        n_cores: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cmd = cmd
        self.filename = filename
        self.n_cores = n_cores
        self.show_progress = show_progress

        self.gamess_scr = gamess_scr.expanduser()
        self.gamess_userscr = gamess_userscr.expanduser()

        self.gamess_options: Dict[str, Any] = {
            "cmd": self.cmd,
            "scr": self.scr,
            "gamess_scr": gamess_scr.expanduser(),
            "gamess_userscr": gamess_userscr.expanduser(),
            "filename": self.filename,
        }

        self._health_check()

    def _health_check(self) -> None:
        assert shell.command_exists(self.cmd), f"{self.cmd} was not found"

    # DEPRECATED
    # def _init_options(self):
    #
    #     self.options = dict()
    #     self.options["basis"] = {
    #         "gbasis": f"{self.method}",
    #     }
    #     self.options["contrl"] = {
    #         "scftyp": "rhf",
    #     }
    #
    #     if self.solvent is not None:
    #         self.options["pcm"] = {
    #             "solvnt": f"{self.solvent}",
    #             "mxts": 15000,
    #             "icav": 1,
    #             "idisp": 1,
    #         }
    #         self.options["tescav"] = {"mthall": 4, "ntsall": 60}
    #
    #     return

    # def _generate_options(self, optimize:bool=True, hessian:bool=False, gradient:bool=False)->dict:
    #
    #     if optimize:
    #         calculation = "optimize"
    #     elif hessian:
    #         calculation = "hessian"
    #     else:
    #         calculation = "energy"
    #
    #     options = dict()
    #     options["contrl"] = {
    #         "runtyp": f"{calculation}",
    #     }
    #
    #     if optimize:
    #         options["statpt"] = {
    #             "opttol": 0.005,
    #             "nstep": 300,
    #             "projct": False,
    #         }
    #
    #     return options

    def calculate(self, molobj: Mol, options: dict) -> List[Optional[dict]]:
        """ """

        # TODO Parallel wrapper

        # Merge options
        # options_prime = dict(ChainMap(options, self.options))
        options_prime = dict(ChainMap(options, {}))

        if "contrl" not in options_prime:
            options_prime["contrl"] = dict()

        options_prime["contrl"]["icharg"] = GAMESS_KEYWORD_CHARGE

        properties_list = []
        n_confs = molobj.GetNumConformers()

        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

        for conf_idx in range(n_confs):

            coord = chembridge.get_coordinates(molobj, confid=conf_idx)
            properties = properties_from_axyzc(
                atoms, coord, charge, options_prime, options_gamess=self.gamess_options
            )

            properties_list.append(properties)

        return properties_list

    def __repr__(self) -> str:
        return f"GamessCalc(cmd={self.cmd},scr={self.scr},gamess_scr={self.gamess_scr},gamess_userscr={self.gamess_userscr})"


def properties_from_axyzc(
    atoms: Union[List[str], np.ndarray],
    coords: np.ndarray,
    charge: int,
    options: dict,
    options_gamess: dict = {},
) -> Optional[dict]:
    """ """

    # Prepare input
    header = get_header(options)

    # set charge
    header = header.format(charge=charge)

    inptxt = get_input(atoms, coords, header)

    # Call GAMESS
    stdout, _ = run_gamess(inptxt, **options_gamess)

    assert stdout is not None, "Uncaught exception"
    # TODO Check stderr

    properties = get_properties(stdout.split("\n"))

    return properties


def prepare_atoms(atoms: Union[List[str], np.ndarray], coordinates: np.ndarray) -> str:

    lines = []
    line = "{:2s}    {:2.1f}    {:f}     {:f}    {:f}"

    for atom, coord in zip(atoms, coordinates):
        iat = chembridge.get_atom_int(atom)
        lines.append(line.format(atom, iat, *coord))

    lines = [" $data", "Title", "C1"] + lines + [" $end"]

    return "\n".join(lines)


def get_input(atoms: Union[List[str], np.ndarray], coords: np.ndarray, header: str) -> str:

    lines = header

    if lines[-1] != "\n":
        lines += "\n"

    lines += prepare_atoms(atoms, coords)

    return lines


def get_header(options: dict) -> str:
    sections = []
    for section_name in options:
        sections.append(get_section(section_name, options[section_name]))
    txt = "\n".join(sections)
    return txt


def get_section(section_name: str, options: dict) -> str:
    section = f" ${section_name} "

    for key, val in options.items():
        if isinstance(val, bool):
            val = ".T." if val else ".F."

        section += f"{key}={val} "

    section += "$end"
    return section


def run_gamess(
    input_text: str,
    cmd: str = GAMESS_CMD,
    scr: Path = constants.SCR,
    filename: Optional[str] = None,
    gamess_scr: Path = GAMESS_SCR,
    gamess_userscr: Path = GAMESS_USERSCR,
    post_clean: bool = True,
    pre_clean: bool = True,
) -> Tuple[str, str]:
    """"""

    # important! Gamess is super sensitive to filename, because it will create
    # filenames in userscr filename in this function should be using tempfile,
    # if filename is None

    if filename is None:
        # filename = hashlib.md5(input_text.encode()).hexdigest() + ".inp"
        filename = next(random_names)

    assert filename is not None

    assert shell.command_exists(cmd), f"Could not find {cmd} in your enviroment"

    if not filename.endswith(".inp"):
        filename += ".inp"

    if pre_clean:
        clean(gamess_scr, filename)
        if gamess_scr != gamess_userscr:
            clean(gamess_userscr, filename)

    full_filename = os.path.join(scr, filename)

    with open(full_filename, "w") as f:
        f.write(input_text)

    command = [cmd, filename]
    command_ = " ".join(command)

    _logger.debug(f"{scr} {command_}")

    stdout: str
    stderr: str

    stdout_, stderr_ = shell.execute(command_, cwd=scr)

    if stdout_ is not None:
        stdout = stdout_
    else:
        stdout = ""

    if stderr_ is not None:
        stderr = stderr_
    else:
        stderr = ""

    if post_clean:
        clean(gamess_scr, filename)
        if gamess_scr != gamess_userscr:
            clean(gamess_userscr, filename)

    return stdout, stderr


def clean(scr: Path, filename: str) -> None:

    _logger.debug(f"removing {scr} {filename}")

    scr = scr.expanduser().absolute()

    files_ = Path(scr).expanduser().glob(filename.replace(".inp", "*"))

    files = list(files_)

    for f in files:
        print(f)
        os.remove(f)


def check_output(output: List[str]) -> bool:
    raise NotImplementedError

    # TODO ELECTRONS, WITH CHARGE ICHARG=

    # TODO redo in Python. Real categories of fail. Better output
    # TODO Did gasphase or solvent phase not converge?
    # TODO How many steps?
    #
    # grep  "FAILURE" *.log
    # grep  "Failed" *.log
    # grep  "ABNORMALLY" *.log
    # grep  "SCF IS UNCONVERGED" *.log
    # grep "resubmit" *.log
    # grep "IMAGINARY FREQUENCY VIBRATION" *.log


def get_errors(lines: List[str]) -> Optional[Dict[str, str]]:
    """
    ddikick.x: Execution terminated due to error(s)
    """

    msg = {}

    safeword = "NSERCH"

    idx = linesio.get_rev_index(lines, safeword)
    has_safeword = idx is not None

    if has_safeword:
        return None

    line: Union[str, List[str]]

    key = "CHECK YOUR INPUT CHARGE AND MULTIPLICITY"
    idx = linesio.get_rev_index(lines, key, stoppattern=safeword)
    if idx is not None:
        line = lines[idx + 1 : idx + 2]
        line = [x.strip().lower().capitalize() for x in line]
        line_ = ". ".join(line)
        line_ = line_.replace("icharg=", "").replace("mult=", "")
        msg["error"] = line_ + ". Only multiplicity 1 allowed."
        _logger.error(line)
        return msg

    key = "ERROR"
    idx = linesio.get_index(lines, key, stoppattern=safeword)
    if idx is not None:
        line = lines[idx]
        line = line.replace("***", "").strip()
        msg["error"] = line
        _logger.error(line)
        return msg

    idx = linesio.get_rev_index(
        lines,
        "FAILURE TO LOCATE STATIONARY POINT, TOO MANY STEPS TAKEN",
        stoppattern=safeword,
    )

    if idx is not None:
        msg["error"] = "TOO_MANY_STEPS"
        _logger.error("Optimization failed. Too many steps.")
        return msg

    idx = linesio.get_rev_index(
        lines,
        "FAILURE TO LOCATE STATIONARY POINT, SCF HAS NOT CONVERGED",
        stoppattern=safeword,
    )

    if idx is not None:
        msg["error"] = "SCF_UNCONVERGED"
        _logger.error("SCF unconverged")
        return msg

    idx = linesio.get_rev_index(
        lines,
        "Error changing to scratch directory",
        stoppattern=safeword,
    )

    if idx is not None:
        msg["error"] = "GAMESS configuration error"
        _logger.error("GAMESS Scratch directory is bad")
        return msg

    idx = linesio.get_rev_index(
        lines,
        "Please save, rename, or erase these files from a previous run",
        stoppattern=safeword,
    )
    if idx is not None:
        msg["error"] = "GAMESS Error. Previous undeleted calculations in scratch folder"
        _logger.error("Previous undeleted calculations in scratch folder with same name")
        return msg

    return msg


def get_properties(lines: List[str], options: dict = {}) -> Optional[dict]:
    """
    Read GAMESS output based on calculation options
    """

    # TODO Better keywords
    # TODO Make a reader list?
    # TODO Move SQM Specific properties
    # TODO Solvation

    reader = None

    runtyp = read_type(lines)

    if runtyp is None:

        errors = get_errors(lines)
        if errors is not None:
            return None

        return None

    method = read_method(lines)
    is_solvation = read_solvation(lines)

    _logger.debug(f"parseing {runtyp} {method} solvation={is_solvation}")

    if method == "HF":
        reader = get_properties_orbitals

    if runtyp == "optimize":
        reader = get_properties_coordinates

    elif runtyp == "hessian":
        reader = get_properties_vibration

    elif is_solvation:
        reader = get_properties_solvation

    if reader is None:
        raise ValueError("No properties to read in GAMESS log")

    properties = reader(lines)

    return properties


def has_failed(lines: List[str]) -> bool:

    msg = "Execution terminated due to error"
    idx = linesio.get_rev_index(lines, msg, stoppattern="TOTAL WALL TIME")
    if idx is not None:
        return True

    return False


def read_solvation(lines: List[str]) -> bool:

    keyword = "INPUT FOR PCM SOLVATION CALCULATION"
    stoppattern = "ELECTRON INTEGRALS"

    idx = linesio.get_index(lines, keyword, stoppattern=stoppattern)
    if idx is not None:
        return True

    return False


def read_type(lines: List[str]) -> Optional[str]:

    idx = linesio.get_index(lines, "CONTRL OPTIONS")

    line: Union[List[str], str]

    if idx is None:
        return None

    idx += 2
    line = lines[idx]

    line = line.split()
    line = line[1]
    line = line.split("=")
    runtyp = line[-1]
    runtyp = runtyp.lower()

    return runtyp


def read_method(lines: List[str]) -> Optional[str]:

    # TODO Add disperion reader

    idx = linesio.get_index(lines, "BASIS OPTIONS")

    if idx is None:
        return None

    line: Union[List[str], str]

    idx += 2
    line = lines[idx]

    line = line.strip().split()
    line = line[0]
    line = line.split("=")
    basis = line[-1]
    basis = basis.lower()

    if basis.upper() in GAMESS_SQM_METHODS:
        method = "SQM"
    else:
        method = "HF"

    return method


def get_properties_coordinates(lines: List[str]) -> dict:

    properties: Dict[str, Any] = {}

    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    if idx is None:
        return {}

    line: Union[List[str], str]

    line = lines[idx]
    line = line.split("=")
    n_atoms = int(line[-1])

    idx = linesio.get_rev_index(
        lines,
        "FAILURE TO LOCATE STATIONARY POINT, TOO MANY STEPS TAKEN",
        stoppattern="NSEARCH",
    )

    if idx is not None:
        properties["error"] = "TOO_MANY_STEPS"
        properties[constants.COLUMN_COORDINATES] = None
        properties[constants.COLUMN_ATOMS] = None
        return properties

    idx = linesio.get_rev_index(
        lines,
        "FAILURE TO LOCATE STATIONARY POINT, SCF HAS NOT CONVERGED",
        stoppattern="NESERCH",
    )

    if idx is not None:
        properties["error"] = "SCF_UNCONVERGED"
        properties[constants.COLUMN_COORDINATES] = None
        properties[constants.COLUMN_ATOMS] = None
        return properties

    idx = linesio.get_rev_index(lines, "EQUILIBRIUM GEOMETRY LOCATED")
    assert idx is not None, "Uncaught error in GAMESS reading"
    idx += 4

    coordinates = np.zeros((n_atoms, 3))
    atoms = np.zeros(n_atoms, dtype=int)

    for i in range(n_atoms):
        line = lines[idx + i]
        line = line.split()
        atom: Union[str, int] = line[1].replace(".0", "")
        atom = int(atom)
        x = line[2]
        y = line[3]
        z = line[4]

        atoms[i] = atom
        coordinates[i][0] = x
        coordinates[i][1] = y
        coordinates[i][2] = z

    idx = linesio.get_rev_index(lines, "HEAT OF FORMATION IS")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx]
    line = line.split()

    # energy in kcal/mol
    hof = float(line[4])

    properties[constants.COLUMN_ATOMS] = atoms
    properties[constants.COLUMN_COORDINATES] = coordinates
    properties[constants.COLUMN_ENERGY] = hof

    return properties


def get_properties_vibration(lines: List[str]) -> dict:

    properties: Dict[str, Any] = {}

    line: Union[List[str], str]

    idx = linesio.get_rev_index(
        lines, "SCF DOES NOT CONVERGE AT VIB", stoppattern="END OF PROPERTY EVALUATION"
    )

    if idx is not None:
        properties["error"] = "Unable to vibrate structure"
        _logger.warning("vibration_unconverged")
        return properties

    # Get number of atoms
    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx]
    line = line.split("=")

    # Get heat of formation
    idx = linesio.get_rev_index(lines, "HEAT OF FORMATION IS")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx]
    line = line.split()

    # energy in kcal/mol
    hof = float(line[4])

    # Check linear
    idx = linesio.get_index(lines, "THIS MOLECULE IS RECOGNIZED AS BEING LINEAR")

    is_linear = idx is not None

    # thermodynamic
    idx = linesio.get_rev_index(lines, "KJ/MOL    KJ/MOL    KJ/MOL   J/MOL-K")
    assert idx is not None, "Uncaught GAMESS error"
    idx += 1
    values = np.zeros((5, 6))

    for i in range(5):
        line = lines[idx + i]
        line = line.split()
        line = line[1:]
        linef = [float(x) for x in line]
        values[i, :] = linef

    # Get Vibrations
    idx_start = linesio.get_rev_index(lines, "FREQ(CM**-1)")
    idx_end = linesio.get_rev_index(lines, "THERMOCHEMISTRY AT T=  298.15 K")
    assert idx_start is not None, "Uncaught GAMESS error"
    assert idx_end is not None, "Uncaught GAMESS error"
    idx_start += 1
    idx_end -= 2
    vibrations = []
    intensities = []
    for i in range(idx_start, idx_end):
        line = lines[i]
        line = line.split()
        freq = float(line[1])
        inte = float(line[-1])
        vibrations.append(freq)
        intensities.append(inte)

    # Cut and save vibration string for jsmol
    # based on number of vibrations and number of atoms
    idx = linesio.get_rev_index(lines, " TAKEN AS ROTATIONS AND TRANSLATIONS.")
    vib_lines = "\n".join(lines[idx:idx_start])

    idx_end = linesio.get_index(lines, "ELECTRON INTEGRALS")
    head_lines = "\n".join(lines[18:idx_end])

    # TODO Make custom molcalc readers
    properties["jsmol"] = head_lines + vib_lines
    properties["linear"] = is_linear
    properties["freq"] = np.array(vibrations)
    properties["intens"] = np.array(intensities)
    properties[COLUMN_THERMO] = values
    properties[constants.COLUMN_ENERGY] = hof

    return properties


def get_properties_orbitals(lines: List[str]) -> dict:

    properties: Dict[str, Any] = {}
    line: Union[List[str], str]

    # Get number of atoms
    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx]
    line = line.split("=")

    idx_start = linesio.get_index(lines, "EIGENVECTORS")
    assert idx_start is not None, "Uncaught GAMESS error"
    idx_start += 4
    idx_end = linesio.get_index(lines, "END OF RHF CALCULATION")
    assert idx_end is not None, "Uncaught GAMESS error"

    energies = []

    wait = False
    j = idx_start
    while j < idx_end:

        line = lines[j].strip()

        if wait:
            if line == "":
                j += 1
                wait = False

        else:
            wait = True

            line = line.split()
            linef = [float(x) for x in line]
            energies += linef

        j += 1

    properties["orbitals"] = np.array(energies)
    properties["stdout"] = "\n".join(lines)

    return properties


def get_properties_solvation(lines: List[str]) -> dict:

    properties: Dict[str, Any] = {}
    line: Union[List[str], str]

    # Check for common errors
    if has_failed(lines):
        errormsg = "ERROR"
        idx = linesio.get_index(lines, errormsg)
        if idx is not None:
            error = lines[idx]
            properties["error"] = error

    # Get number of atoms
    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx]
    line = line.split("=")
    n_atoms = int(line[-1])

    # Get solvation data,charge of molecule, surface area, dipole

    idx = linesio.get_rev_index(lines, "ELECTROSTATIC INTERACTION")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx]
    line = line.split()
    electrostatic_interaction = float(line[-2])

    line = lines[idx + 1].split()
    pierotti_cavitation_energy = float(line[-2])

    line = lines[idx + 2].split()
    dispersion_free_energy = float(line[-2])

    line = lines[idx + 3].split()
    repulsion_free_energy = float(line[-2])

    line = lines[idx + 4].split()
    total_interaction = float(line[-2])

    total_non_polar = pierotti_cavitation_energy + dispersion_free_energy + repulsion_free_energy

    idx = linesio.get_index(lines, "CHARGE OF MOLECULE")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx]
    line = line.split("=")
    charge = int(line[-1])

    idx = linesio.get_rev_index(lines, "SURFACE AREA")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx]
    line = line.split()
    surface_area_ = line[2]
    surface_area = float(surface_area_.replace("(A**2)", ""))

    idx = linesio.get_rev_index(lines, "DEBYE")
    assert idx is not None, "Uncaught GAMESS error"
    line = lines[idx + 1]
    line = line.split()
    linef = [float(x) for x in line]
    dxyz = linef[0:3]
    dtot = linef[-1]

    idx = linesio.get_rev_index(lines, "MOPAC CHARGES")
    assert idx is not None, "Uncaught GAMESS error"
    idx += 3
    partial_charges = np.zeros(n_atoms)
    for i in range(n_atoms):
        line = lines[idx + i]
        line = line.split()
        atom_charge = float(line[-2])
        partial_charges[i] = atom_charge

    properties["charges"] = partial_charges
    properties["solvation_total"] = total_interaction
    properties["solvation_polar"] = electrostatic_interaction
    properties["solvation_nonpolar"] = total_non_polar
    properties["surface"] = surface_area
    properties["total_charge"] = charge
    properties["dipole"] = np.array(dxyz)
    properties["dipole_total"] = dtot

    return properties
