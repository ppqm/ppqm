import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
from cclib.io.ccio import ccframe
from cclib.parser import MOPAC as MOPACParser  # type: ignore

from ppqm import constants
from ppqm.calculators.mopac.constants import MOPAC_INPUT_EXTENSION, MOPAC_OUTPUT_EXTENSION
from ppqm.utils import linesio

_logger = logging.getLogger(__name__)

# NOTE
# Should be possible to get graph of molecule using
# keyword "local" on mopac
# http://openmopac.net/manual/localize.html

# NOTE
# There is an internal check for distances, but can be ignored with
# "GEO-OK" keywords.


def read_chunks(file: io.TextIOWrapper) -> Generator[io.StringIO, None, None]:
    """Try to read a MOPAC output with multiple molecules. Break the stream in each Finish"""

    lines = ""
    n = 0
    for line in file:

        print(line)

        lines += "line"
        n += 1

        if "TOTAL JOB TIME" in line:
            yield io.StringIO(lines)
            break

        if "CALCULATION RESULTS" in line and n > 20:
            yield io.StringIO(lines)
            lines = ""
            n = 0

    return


def read_properties(filename: str | Path):

    # TODO Iterator, returns

    return


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


def has_error(lines: list[str]) -> bool:
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


def get_properties(lines: list[str], use_cclib: bool = True) -> dict | None:
    """
    TODO Check common errors

    Errors detected in keywords.
    UNRECOGNIZED KEY-WORDS: (MULIKEN)

    """

    if use_cclib:

        # Make a IO stream
        _stream = io.StringIO("\n".join(lines))
        mopac_parser = MOPACParser(_stream)
        data = mopac_parser.parse()

        # parser = ccopen(_stream)
        # data = parser.parse()

        print(data.natom)
        print(data.nelectrons)

        print(mopac_parser)

        pdf = ccframe([data])

        print(pdf)
        print(pdf.columns)

        print(data.metadata)

        assert False

        # TOOD Get properties
        return {}

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
