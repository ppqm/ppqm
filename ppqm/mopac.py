import logging
import os
from collections import ChainMap

import numpy as np

from . import chembridge, constants, linesio, shell
from .calculator import BaseCalculator

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

_logger = logging.getLogger("mopac")


class MopacCalculator(BaseCalculator):
    """"""

    def __init__(
        self,
        cmd=MOPAC_CMD,
        filename=MOPAC_FILENAME,
        scr=constants.SCR,
        options=MOPAC_DEFAULT_OPTIONS,
        n_cores=None,
        show_progress=False,
    ):
        """"""

        super().__init__(scr=scr)

        self.cmd = cmd

        # Constants
        self.filename = filename

        # Default calculate options
        self.options = options

        return

    def calculate(self, molobj, options):

        # Merge options
        options_prime = ChainMap(options, self.options)
        options_prime = dict(options_prime)
        options_prime["charge"] = MOPAC_KEYWORD_CHARGE

        input_string = self._get_input_str(molobj, options_prime, opt_flag=True)

        filename = self.scr / self.filename

        with open(filename, "w") as f:
            f.write(input_string)

        _logger.debug(f"{self.scr} {self.filename} {self.cmd}")
        # Run mopac
        self._run_file()

        calculations = self._read_file()
        results = [get_properties(output_lines) for output_lines in calculations]

        return results

    def _generate_options(self, optimize=True, hessian=False, gradient=False):
        """ Generate options for calculation types """

        if optimize:
            calculation = "opt"
        elif hessian:
            calculation = "hessian"
        else:
            calculation = "1scf"

        options = dict()
        options[calculation] = None

        return options

    def _get_input_str(self, molobj, options, title="", opt_flag=False):
        """
        Create MOPAC input string from molobj
        """

        n_confs = molobj.GetNumConformers()

        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)
        header = get_header(options)

        txt = []
        for i in range(n_confs):

            coord = chembridge.get_coordinates(molobj, confid=i)
            header_prime = header.format(charge=charge, title=f"{title}_Conf_{i}")
            tx = get_input(atoms, coord, header_prime, opt_flag=opt_flag)
            txt.append(tx)

        txt = "".join(txt)

        return txt

    def _run_file(self):

        runcmd = f"{self.cmd} {self.filename}"

        stdout, stderr = shell.execute(runcmd, cwd=self.scr)

        # TODO Check stderr and stdout

        return

    def _read_file(self):

        filename = self.scr / self.filename
        filename = str(filename)
        filename = filename.replace(".mop", ".out")

        with open(filename, "r") as f:
            lines = f.readlines()

        # Check for erros
        if has_error(lines):
            return []

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
        this = f"MopacCalc(scr={self.scr}, cmd={self.cmd})"
        return this


def run_mopac(filename, cmd=MOPAC_CMD, scr=None):
    """ Run mopac on filename, inside scr directory"""

    command = [cmd, filename]
    command = " ".join(command)

    stdout, stderr = shell.execute(command, cwd=scr)

    # TODO Check stdout and stderr for error and return False

    return True


def get_header(options):
    """ return mopac header from options dict """

    title = options.get("title", "TITLE")
    if "title" in options:
        del options["title"]

    header = [""] * 3
    header[1] = title
    header[0] = list()

    for key, val in options.items():

        if val is not None:
            keyword = f"{key}={val}"
        else:
            keyword = f"{key}"

        header[0].append(keyword)

    header[0] = " ".join(header[0])
    header = "\n".join(header)

    return header


def get_input(atoms, coords, header, opt_flag=False):
    """ Generate input text for MOPAC calculation """

    flag: int = 1 if opt_flag else 0

    txt = header
    txt += "\n"

    for atom, coord in zip(atoms, coords):
        line = MOPAC_ATOMLINE.format(atom=atom, x=coord[0], y=coord[1], z=coord[2], opt_flag=flag)

        txt += line + "\n"

    txt += "\n"

    return txt


def properties_from_axyzc(atoms, coords, charge, header, **kwargs):
    """ Calculate properties for atoms, coord and charge  """

    properties_list = properties_from_many_axyzc([atoms], [coords], [charge], header, **kwargs)

    properties = properties_list[0]

    return properties


def properties_from_many_axyzc(
    atoms_list,
    coords_list,
    charge_list,
    header,
    titles=None,
    optimize=False,
    cmd=MOPAC_CMD,
    filename=MOPAC_FILENAME,
    scr=None,
):
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

    input_texts = "".join(input_texts)

    if scr is None:
        scr = ""

    # Save file
    with open(os.path.join(scr, filename), "w") as f:
        f.write(input_texts)

    # Run file
    run_mopac(filename, scr=scr)

    # Return properties
    properties_list = []

    for lines in read_output(filename, scr=scr):
        properties = get_properties(lines)
        properties_list.append(properties)

    return properties_list


def read_output(filename, scr=None, translate_filename=True):

    if scr is None:
        scr = ""

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


def has_error(lines):
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

        if not idx:
            continue

        msg = lines[idx]
        msg = msg.replace("*", "")
        msg = msg.strip()
        _logger.error(msg)

    if any(idxs):
        return True

    return False


def get_properties(lines):
    """
    TODO Check common errors

    Errors detected in keywords.
    UNRECOGNIZED KEY-WORDS: (MULIKEN)

    """

    if is_1scf(lines):
        properties = get_properties_1scf(lines)

    else:
        properties = get_properties_optimize(lines)

    return properties


def is_1scf(lines):
    """

    Check if output is a single point or optimization

    """

    keyword = "1SCF WAS USED"
    stoppattern = "CYCLE"

    idx = linesio.get_indices(lines, keyword, stoppattern=stoppattern)

    if idx is None or len(idx) == 0:
        return False

    return True


def get_properties_optimize(lines):
    """"""

    properties = {}

    # Enthalpy of formation
    idx_hof = linesio.get_rev_index(lines, "FINAL HEAT OF FORMATION")

    if idx_hof is None:
        value = float("nan")
    else:
        line = lines[idx_hof]
        line = line.split("FORMATION =")
        line = line[1]
        line = line.split()
        value = line[0]
        value = float(value)

    # enthalpy of formation in kcal/mol
    properties["h"] = value

    # optimized coordinates
    i = linesio.get_rev_index(lines, "CARTESIAN")

    line = lines[i]
    if i is not None and "ATOM LIST" in line:
        i = None

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
            xyz = [x, y, z]
            xyz = [float(c) for c in xyz]
            coord.append(xyz)
            j += 1

        coord = np.array(coord)

    properties[constants.COLUMN_COORDINATES] = coord
    properties[constants.COLUMN_ATOMS] = symbols

    return properties


def get_properties_1scf(lines):
    """"""

    properties = {}

    # Enthalpy of formation
    idx_hof = linesio.get_rev_index(lines, "FINAL HEAT OF FORMATION")
    line = lines[idx_hof]
    line = line.split("FORMATION =")
    line = line[1]
    line = line.split()
    value = line[0]
    value = float(value)

    # enthalpy in kcal/mol
    properties["h"] = value

    return properties
