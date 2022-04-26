import logging
from collections import ChainMap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm  # type: ignore[import]

from ppqm import chembridge, constants, units
from ppqm.calculator import BaseCalculator
from ppqm.chembridge import Mol
from ppqm.utils import WorkDir, linesio, shell

G16_CMD = "g16"
G16_FILENAME = "_tmp_g16_input.com"

_logger = logging.getLogger(__name__)

COLUMN_SCF_ENERGY = "scf_energy"
COLUMN_CONVERGED = "is_converged"
COLUMN_MULIKEN_CHARGES = "mulliken charges"
COLUMN_CM5_CHARGES = "cm5_charges"
COLUMN_HIRSHFELD_CHARGES = "hirshfeld_charges"
COLUMN_NBO_BONDORDER = "bond_orders"
COLUMN_SHIELDING_CONSTANTS = "shielding_constants"


class GaussianCalculator(BaseCalculator):
    def __init__(
        self,
        cmd: str = G16_CMD,
        filename: str = G16_FILENAME,
        show_progress: bool = False,
        n_cores: int = 1,
        memory: int = 2,
        **kwargs: Any,
    ):

        super().__init__(**kwargs)

        self.cmd = cmd
        self.filename = filename
        self.n_cores = n_cores
        self.memory = memory
        self.show_progress = show_progress

        self.g16_options: Dict[str, Any] = {
            "cmd": self.cmd,
            "scr": self.scr,
            "filename": self.filename,
            "memory": self.memory,
        }

        self.options: Dict[str, Any] = {}

        self.health_check()

    def __repr__(self) -> str:
        return f"G16Calc(cmd={self.cmd}, scr={self.scr}, n_cores={self.n_cores}, memory={self.memory}gb)"

    def health_check(self) -> None:
        """ """
        # TODO Check version

    def calculate(
        self, molobj: Mol, options: dict, footer: Optional[str] = None
    ) -> List[Optional[dict]]:
        """ """

        if self.n_cores > 1:
            raise NotImplementedError("Parallel not implemented yet.")
        else:
            results = self.calculate_serial(molobj, options, footer=footer)

        return results

    def calculate_serial(
        self, molobj: Mol, options: dict, footer: Optional[str] = None
    ) -> List[Optional[dict]]:
        """ """

        # If not singlet "spin" is part of options
        if "spin" in options.keys():
            spin = int(options.pop("spin"))
        else:
            spin = int(1)

        options_prime = dict(ChainMap(options, self.options))

        n_confs = molobj.GetNumConformers()
        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

        pbar = None
        if self.show_progress:
            pbar = tqdm(
                total=n_confs,
                desc="g16(1)",
                **constants.TQDM_OPTIONS,
            )

        properties_list = []
        for conf_idx in range(n_confs):

            coord = chembridge.get_coordinates(molobj, confid=conf_idx)

            properties = get_properties_from_axyzc(
                atoms,
                coord,
                charge,
                spin,
                options=options_prime,
                footer=footer,
                **self.g16_options,
            )

            properties_list.append(properties)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return properties_list


def get_properties_from_axyzc(
    atoms_str: Union[List[str], np.ndarray],
    coordinates: np.ndarray,
    charge: int,
    spin: int,
    options: dict = {},
    scr: Path = constants.SCR,
    cmd: str = G16_CMD,
    filename: str = G16_FILENAME,
    footer: Optional[str] = None,
    keep_files: bool = False,
    memory: int = 2,
) -> Optional[dict]:

    if not filename.endswith(".com"):
        filename += ".com"

    workdir = WorkDir(dir=scr, prefix="g16_", keep=keep_files)
    scr = workdir.get_path()

    # write input file
    input_header = get_header(options, memory=memory)

    inputstr = get_inputfile(
        atoms_str, coordinates, charge, spin, input_header, footer=footer, title="g16 input"
    )

    with open(scr / filename, "w") as f:
        f.write(inputstr)

    # Run subprocess cmd
    cmd = " ".join([cmd, filename])
    _logger.debug(cmd)

    lines = list(shell.stream(cmd, cwd=scr))

    termination_pattern = "Normal termination of Gaussian"
    idx = linesio.get_rev_index(lines, termination_pattern, stoppattern="File lengths")
    if idx is None:
        _logger.critical("Abnormal termination of Gaussian")
        return None

    # Parse properties from Gaussian output
    properties = read_properties(lines, options)

    return properties


def get_inputfile(
    atom_strs: Union[List[str], np.ndarray],
    coordinates: np.ndarray,
    charge: int,
    spin: int,
    header: str,
    footer: Optional[str] = None,
    title: str = "title",
) -> str:
    """ """

    inputstr = header + 2 * "\n"
    inputstr += f"  {title}" + 2 * "\n"

    inputstr += f"{charge}  {spin} \n"
    for atom_str, coord in zip(atom_strs, coordinates):
        inputstr += f"{atom_str}  " + " ".join([str(x) for x in coord]) + "\n"
    inputstr += "\n"  # magic line

    if footer is not None:
        inputstr += footer + "\n"

    return inputstr


def get_header(options: dict, memory: int = 2) -> str:
    """Write G16 header"""

    header = f"%mem={memory}gb\n"
    header += "# "
    for key, value in options.items():
        if (value is None) or (not value):
            header += f"{key} "
        elif isinstance(value, str):
            header += f"{key}={value} "
        else:
            header += f"{key}=("
            for subkey, subvalue in value.items():
                if (subvalue is None) or (not subvalue):
                    header += f"{subkey}, "
                else:
                    header += f"{subkey}={subvalue}, "
            header = header[:-2] + ") "

    return header


def read_properties(lines: List[str], options: dict) -> Optional[dict]:
    """Extract values from output depending on calculation options"""

    # Collect readers
    readers: List[Callable] = []

    if "opt" in options:
        raise NotImplementedError("not implemented opt properties parser")
        # reader = read_properties_opt
    else:
        readers.append(read_properties_sp)

    if "pop" in options:
        if "nboread" in options["pop"]:
            readers.append(get_nbo_bond_orders)

        if "hirshfeld" in options["pop"]:
            readers.append(get_hirsfeld_charges)
            readers.append(get_cm5_charges)

    if "nmr" in options:
        readers.append(get_nmr_shielding_constants)

    # Get properties
    properties = dict()
    for reader in readers:
        new_properties = reader(lines)
        assert isinstance(new_properties, dict)

        properties.update(new_properties)

    return properties


def read_properties_sp(lines: List[str]) -> Optional[dict]:
    """Read Mulliken charges"""
    properties = dict()

    for line in lines:
        if "SCF Done:  E(" in line:
            scf_energy = float(line.split()[4]) * units.hartree_to_kcalmol
            properties[COLUMN_SCF_ENERGY] = scf_energy
            break

    charges = get_mulliken_charges(lines)

    if charges is not None:
        properties.update(charges)

    return properties


def read_properties_opt(lines: List[str]) -> dict:
    """ """
    raise NotImplementedError


def get_mulliken_charges(lines: List[str]) -> Optional[dict]:
    """Read Mulliken charges"""
    keywords = ["Mulliken charges:", "Sum of Mulliken charges"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)
    if start is None or stop is None:
        return None
    mulliken_charges = [float(x.split()[-1]) for x in lines[start + 2 : stop]]
    return {COLUMN_MULIKEN_CHARGES: mulliken_charges}


def get_hirsfeld_charges(lines: List[str]) -> Optional[dict]:
    """Read Hirsfeld charges - run a NBO calculation"""
    keywords = ["Hirshfeld charges,", "Hirshfeld charges with"]
    start, stop = linesio.get_indices_patterns(lines, keywords)
    if start is None or stop is None:
        return None
    hirshfeld_charges = [float(line.split()[2]) for line in lines[start + 2 : stop - 1]]

    return {COLUMN_HIRSHFELD_CHARGES: hirshfeld_charges}


def get_cm5_charges(lines: List[str]) -> Optional[dict]:
    """Read CM5 charges - run a NBO calculation"""
    keywords = ["Hirshfeld charges,", "Hirshfeld charges with"]
    start, stop = linesio.get_indices_patterns(lines, keywords)
    if start is None or stop is None:
        return None
    cm5_charges = [float(line.split()[-1]) for line in lines[start + 2 : stop - 1]]

    return {COLUMN_CM5_CHARGES: cm5_charges}


def get_nbo_bond_orders(lines: List[str]) -> Optional[dict]:
    """
    Read Wiberg index - run a NBOread calculation.
    N.B. add $nbo bndidx $end to the footer.
    """

    keywords = ["Wiberg bond index matrix", "Wiberg bond index"]
    start, stop = linesio.get_indices_patterns(lines, keywords)

    if start is None:
        _logger.critical("You did not add: '$nbo bndidx $end' to the footer")
        return None

    # Extract Bond order matrix
    bond_idx_blocks: List[List[List[float]]] = []
    block_idx = 0
    for line in lines[start + 1 : stop]:
        line_ = line.strip().split()
        if len(line_):  # filter empty strings
            if "Atom" in line_:
                continue

            if set("".join(line)).issubset(set("-")):
                bond_idx_blocks.append([])
                block_idx += 1
                continue

            bond_idx_blocks[block_idx - 1].append([float(x) for x in line_[2:]])

    # Format matrix
    bond_order_matrix = bond_idx_blocks[0]
    for bond_idx_block in bond_idx_blocks[1:]:
        for i, bond_idx in enumerate(bond_idx_block):
            bond_order_matrix[i].extend(bond_idx)

    return {COLUMN_NBO_BONDORDER: bond_order_matrix}


def get_nmr_shielding_constants(lines: List[str]) -> dict:
    """Read GIAO NMR shielding constants"""
    keywords = ["Magnetic shielding tensor (ppm):", "************"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)

    shielding_constants = []
    for line in lines[start:stop]:
        if "Isotropic" in line:
            shielding_constants.append(float(line.strip().split()[4]))

    return {COLUMN_SHIELDING_CONSTANTS: shielding_constants}
