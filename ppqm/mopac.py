import logging
import os
from collections import ChainMap
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np

from ppqm import chembridge, constants
from ppqm.calculator import BaseCalculator
from ppqm.chembridge import Mol
from ppqm.utils import linesio, shell

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
MOPAC_INPUT_EXTENSION = "mop"
MOPAC_OUTPUT_EXTENSION = "out"
MOPAC_KEYWORD_CHARGE = "{charge}"
MOPAC_FILENAME = "_tmp_mopac." + MOPAC_INPUT_EXTENSION

MOPAC_DEFAULT_OPTIONS = {"precise": None, "mullik": None}

_logger = logging.getLogger(__name__)


class MopacCalculator(BaseCalculator):
    """"""

    def __init__(
        self,
        cmd: str = MOPAC_CMD,
        filename: str = MOPAC_FILENAME,
        scr: Path = constants.SCR,
        options: dict = MOPAC_DEFAULT_OPTIONS,
        n_cores: int = 1,
        show_progress: bool = False,
    ) -> None:
        """"""

        super().__init__(scr=scr)

        self.cmd = cmd

        # Constants
        self.filename = filename

        # Default calculate options
        self.options = options

        self.n_cores = n_cores
        self.show_progress = show_progress

    def calculate(self, molobj: Mol, options: dict) -> List[Optional[dict]]:

        # Merge options
        options_prime = dict(ChainMap(options, self.options))
        options_prime["charge"] = MOPAC_KEYWORD_CHARGE

        input_string = self._get_input_str(molobj, options_prime, opt_flag=True)

        filename = self.scr / self.filename

        with open(filename, "w") as f:
            f.write(input_string)

        _logger.debug(f"{self.scr} {self.filename} {self.cmd}")
        # Run mopac
        self._run_file()

        calculations = self._read_file()
        results: List[Optional[dict]] = [
            get_properties(output_lines) for output_lines in calculations
        ]

        return results

    def _generate_options(self, optimize: bool = True, hessian: bool = False, gradient: bool = False) -> dict:  # type: ignore[override]
        """Generate options for calculation types"""

        if optimize:
            calculation = "opt"
        elif hessian:
            calculation = "hessian"
        else:
            calculation = "1scf"

        options: Dict[str, Any] = dict()
        options[calculation] = None

        return options

    def _get_input_str(
        self, molobj: Mol, options: dict, title: str = "", opt_flag: bool = False
    ) -> str:
        """Create MOPAC input string from molobj"""

        n_confs = molobj.GetNumConformers()

        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)
        header = get_header(options)

        txt = []
        for i in range(n_confs):

            coord = chembridge.get_coordinates(molobj, confid=i)
            header_prime = header.format(charge=charge, title=f"{title}_Conf_{i}")
            tx = get_input(atoms, coord, header_prime, opt_flag=opt_flag)
            txt.append(tx)

        return "".join(txt)

    def _run_file(self) -> None:

        runcmd = f"{self.cmd} {self.filename}"

        stdout, stderr = shell.execute(runcmd, cwd=self.scr)

        # TODO Check stderr and stdout

        return

    def _read_file(self) -> Generator[List[str], None, None]:

        filename = str(self.scr / self.filename)
        filename = filename.replace(".mop", ".out")

        with open(filename, "r") as f:
            lines = f.readlines()

        # Check for erros
        if has_error(lines):
            return

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

    def __repr__(self) -> str:
        return f"MopacCalc(scr={self.scr}, cmd={self.cmd})"


def run_mopac(filename: str, cmd: str = MOPAC_CMD, scr: Optional[Path] = None) -> bool:
    """Run mopac on filename, inside scr directory"""

    command = " ".join([cmd, filename])

    stdout, stderr = shell.execute(command, cwd=scr)

    # TODO Check stdout and stderr for error and return False

    return True


def get_header(options: dict) -> str:
    """return mopac header from options dict"""

    title = options.get("title", "TITLE")
    if "title" in options:
        del options["title"]

    header: List[Any] = [""] * 3
    header[1] = title
    header[0] = list()

    for key, val in options.items():

        if val is not None:
            keyword = f"{key}={val}"
        else:
            keyword = f"{key}"

        header[0].append(keyword)

    header[0] = " ".join(header[0])
    return "\n".join(header)


def get_input(
    atoms: Union[List[str], np.ndarray], coords: np.ndarray, header: str, opt_flag: bool = False
) -> str:
    """Generate input text for MOPAC calculation"""

    flag: int = 1 if opt_flag else 0

    txt = header
    txt += "\n"

    for atom, coord in zip(atoms, coords):
        line = MOPAC_ATOMLINE.format(atom=atom, x=coord[0], y=coord[1], z=coord[2], opt_flag=flag)

        txt += line + "\n"

    txt += "\n"

    return txt


def properties_from_axyzc(
    atoms: Union[List[str], np.ndarray],
    coords: np.ndarray,
    charge: int,
    header: str,
    **kwargs: Any,
) -> Optional[dict]:
    """Calculate properties for atoms, coord and charge"""

    properties_list = properties_from_many_axyzc([atoms], [coords], [charge], header, **kwargs)

    properties = properties_list[0]

    return properties


def properties_from_many_axyzc(
    atoms_list: List[Union[List[str], np.ndarray]],
    coords_list: List[np.ndarray],
    charge_list: List[int],
    header: str,
    titles: Optional[List[str]] = None,
    optimize: bool = False,
    cmd: str = MOPAC_CMD,
    filename: str = MOPAC_FILENAME,
    scr: Optional[Path] = None,
) -> List[Optional[dict]]:
    """
    Calculate properties from a series of atoms, coord and charges. Written as one input file for MOPAC.

    NOTE: header requires {charge} in string for formatting

    """

    input_texts = list()

    for i, (atoms, coords, charge) in enumerate(zip(atoms_list, coords_list, charge_list)):

        if titles is None:
            title = ""
        else:
            title = titles[i]

        header_prime = header.format(charge=charge, title=title)
        input_text = get_input(atoms, coords, header_prime, opt_flag=optimize)
        input_texts.append(input_text)

    input_texts_ = "".join(input_texts)

    if scr is None:
        scr = Path("")

    # Save file
    with open(scr / filename, "w") as f:
        f.write(input_texts_)

    # Run file
    run_mopac(filename, scr=scr, cmd=cmd)

    # Return properties
    properties_list = []

    # Stream output
    for lines in read_output(filename, scr=scr):
        properties = get_properties(lines)
        properties_list.append(properties)

    return properties_list


def read_output(
    filename: str, scr: Optional[Path] = None, translate_filename: bool = True
) -> Generator[List[str], None, None]:

    if scr is None:
        scr = Path("")

    if translate_filename:
        filename = os.path.join(scr, filename)
        filename = filename.replace("." + MOPAC_INPUT_EXTENSION, "")
        filename += "." + MOPAC_OUTPUT_EXTENSION

    with open(filename, "r") as f:
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


def has_error(lines: List[str]) -> bool:
    #
    #  *  Errors detected in keywords.  Job stopped here to avoid wasting time.
    #  *
    #  *******************************************************************************
    #
    #  ******************************************************************************
    #  *                                                                            *
    #  *     Error and normal termination messages reported in this calculation     *
    #  *                                                                            *
    #  * UNRECOGNIZED KEY-WORDS: (GFN=1 ALPB=WATER)                                 *
    #  * IF THESE ARE DEBUG KEYWORDS, ADD THE KEYWORD "DEBUG".                      *
    #  * JOB ENDED NORMALLY                                                         *
    #  *                                                                            *
    #  ******************************************************************************
    #
    #
    #
    #  TOTAL JOB TIME:             0.00 SECONDS

    keywords = [
        "UNRECOGNIZED",
        "Error",
        "error",
    ]

    idxs = linesio.get_rev_indices_patterns(lines, keywords, maxiter=50)

    for idx in idxs:

        if idx is None:
            continue

        msg = lines[idx]
        msg = msg.replace("*", "")
        msg = msg.strip()
        _logger.error(msg)

    if any(idxs):
        return True

    return False


def get_properties(lines: List[str]) -> Optional[dict]:
    """
    TODO Check common errors

    Errors detected in keywords.
    UNRECOGNIZED KEY-WORDS: (MULIKEN)

    """

    properties: Optional[dict]

    if is_1scf(lines):
        properties = get_properties_1scf(lines)

    else:
        properties = get_properties_optimize(lines)

    return properties


def is_1scf(lines: List[str]) -> bool:
    """

    Check if output is a single point or optimization

    """

    keyword = "1SCF WAS USED"
    stoppattern = "CYCLE"

    idx = linesio.get_indices(lines, keyword, stoppattern=stoppattern)

    if idx is None or len(idx) == 0:
        return False

    return True


def get_properties_optimize(lines: List[str]) -> Optional[dict]:
    """"""

    properties: Dict[str, Any] = {}
    line: Union[List[str], str]

    # Enthalpy of formation
    idx_hof = linesio.get_rev_index(lines, "FINAL HEAT OF FORMATION")

    if idx_hof is None:
        value = float("nan")
    else:
        line = lines[idx_hof]
        line = line.split("FORMATION =")
        line = line[1]
        line = line.split()
        value = float(line[0])

    # enthalpy of formation in kcal/mol
    properties["h"] = value

    # optimized coordinates
    i = linesio.get_rev_index(lines, "CARTESIAN")
    assert i is not None, "Uncaught MOPAC error"

    line = lines[i]
    if i is not None and "ATOM LIST" in line:
        i = None

    coord: Optional[Union[list, np.ndarray]]

    if i is None:
        coord = None
        symbols = None
    else:
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
            line = lines[j].split()

            atm = line[idx_atm]
            symbols.append(atm)

            x = line[idx_x]
            y = line[idx_y]
            z = line[idx_z]
            xyz = [float(c) for c in [x, y, z]]
            coord.append(xyz)
            j += 1

        coord = np.array(coord)

    properties[constants.COLUMN_COORDINATES] = coord
    properties[constants.COLUMN_ATOMS] = symbols

    return properties


def get_properties_1scf(lines: List[str]) -> dict:
    """"""

    properties: Dict[str, Any] = {}
    line: Union[List[str], str]

    # Enthalpy of formation
    idx_hof = linesio.get_rev_index(lines, "FINAL HEAT OF FORMATION")
    assert idx_hof is not None, "Uncaught MOPAC error"
    line = lines[idx_hof]
    line = line.split("FORMATION =")
    line = line[1]
    line = line.split()
    value = float(line[0])

    # enthalpy in kcal/mol
    properties["h"] = value

    return properties
