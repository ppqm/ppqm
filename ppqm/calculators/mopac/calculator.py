import logging
from collections import ChainMap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from cclib.io.ccio import ccframe
from cclib.parser import MOPAC as MOPACParser  # type: ignore

from ppqm import chembridge, constants
from ppqm.calculators import BaseCalculator
from ppqm.calculators.mopac.constants import (
    MOPAC_ATOMLINE,
    MOPAC_CMD,
    MOPAC_DEFAULT_OPTIONS,
    MOPAC_FILENAME,
    MOPAC_KEYWORD_CHARGE,
)
from ppqm.calculators.mopac.parser import get_properties, read_chunks, read_output
from ppqm.chembridge import Mol
from ppqm.utils import shell

# NOTE
# Should be possible to get graph of molecule using
# keyword "local" on mopac
# http://openmopac.net/manual/localize.html

# NOTE
# There is an internal check for distances, but can be ignored with
# "GEO-OK" keywords.

_logger = logging.getLogger(__name__)


class MopacCalculator(BaseCalculator):
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

        results = self._read_file()
        list(results)
        assert False

        # DEPRECATED
        # calculations = self._read_file()
        # results: List[Optional[dict]] = [
        #     get_properties(output_lines) for output_lines in calculations
        # ]

        return results

    def _generate_options(self, optimize: bool = True, hessian: bool = False, gradient: bool = False) -> dict:  # type: ignore[override]
        """Generate options for calculation types"""

        if gradient:
            raise NotImplementedError

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

    def _run_file(self) -> bool:
        status = run_mopac(self.filename, cmd=self.cmd, scr=self.scr)
        return status

    def _read_file(self) -> Any:

        filename = (self.scr / self.filename).with_suffix(".out")

        if not filename.is_file():
            raise FileExistsError(f"Could not find MOPAC output file {filename}")

        f = open(filename, "r")

        datas = []
        for chunk in read_chunks(f):

            mopac_parser = MOPACParser(chunk)
            print(mopac_parser)

            # TODO Send ExitSignal to stream
            data = mopac_parser.parse()
            datas.append(data)

            print(data.metadata)

        f.close()

        pdf = ccframe(datas)

        print(pdf)
        print(pdf.columns)

        assert False
        # return

        # with open(filename, "r") as f:
        #     lines = f.readlines()

        # # Check for erros
        # if has_error(lines):
        #     return

        # molecule_lines = []

        # for line in lines:

        #     molecule_lines.append(line.strip("\n"))

        #     if "TOTAL JOB TIME" in line:
        #         yield molecule_lines
        #         return

        #     if "CALCULATION RESULTS" in line and len(molecule_lines) > 20:
        #         yield molecule_lines
        #         molecule_lines = []

        # return

    def __repr__(self) -> str:
        return f"MopacCalc(scr={self.scr}, cmd={self.cmd})"


def run_mopac(filename: str, cmd: str = MOPAC_CMD, scr: Optional[Path] = None) -> bool:
    """
    Run mopac on filename, inside scr directory

    Args:
        filename: str - Input file
        cmd: str - MOPAC cmd
        scr: str - Scratch directory

    Returns:
        bool - True if mopac was succesfully run

    Raises:
        ValueError - If command was unsuccessful
    """
    command = " ".join([cmd, filename])
    _logger.debug(f"Run mopac {command} in {scr}")
    _, stderr, exit_code = shell.execute(command, cwd=scr)

    if exit_code > 0:
        _logger.error(f"MOPAC run error {exit_code}: {stderr}")
        raise RuntimeError(stderr)

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

    properties_list = properties_from_many_axyzc(
        [list(atoms)], [coords], [charge], header, **kwargs
    )

    properties = properties_list[0]

    return properties


def properties_from_many_axyzc(
    atoms_list: list[list[str]] | np.ndarray,
    coords_list: list[np.ndarray],
    charge_list: list[int],
    header: str,
    titles: List[str] | None = None,
    optimize: bool = False,
    cmd: str = MOPAC_CMD,
    filename: str = MOPAC_FILENAME,
    scr: Path | None = None,
    parser: None = None,
) -> List[dict | None]:
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

    # NOTE MOPAC supports multi-molecule inputfiles
    f = open(scr / filename, "r")

    # Parse file until complete, then continue
    # TODO Create a parser

    # Stream output
    for lines in read_output(filename, scr=scr):
        properties = get_properties(lines)
        properties_list.append(properties)

    f.close()

    return properties_list
