import logging
import pathlib
import tempfile
from collections import ChainMap

from tqdm import tqdm

from ppqm import chembridge, constants, linesio, shell, units
from ppqm.calculator import BaseCalculator

G16_CMD = "g16"
G16_FILENAME = "_tmp_g16_input.com"

_logger = logging.getLogger("g16")

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
        cmd=G16_CMD,
        filename=G16_FILENAME,
        show_progress=False,
        n_cores=None,
        memory=2,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.cmd = cmd
        self.filename = filename
        self.n_cores = n_cores
        self.memory = memory
        self.show_progress = show_progress

        self.g16_options = {
            "cmd": self.cmd,
            "scr": self.scr,
            "filename": self.filename,
            "memory": self.memory,
        }

        # Default G16 options
        self.options = {}

        #
        self.health_check()

    def __repr__(self) -> str:
        return f"G16Calc(cmd={self.cmd}, scr={self.scr}, n_cores={self.n_cores}, memory={self.memory}gb)"

    def health_check(self):
        """ """

    def calculate(self, molobj, options, footer=None):
        """ """

        if self.n_cores > 1:
            raise NotImplementedError("Parallel not implemented yet.")
        else:
            results = self.calculate_serial(molobj, options, footer=footer)

        return results

    def calculate_serial(self, molobj, options, footer=None):
        """ """

        # If not singlet "spin" is part of options
        if "spin" in options.keys():
            spin = str(options.pop("spin"))
        else:
            spin = str(1)

        options_prime = ChainMap(options, self.options)
        options_prime = dict(options_prime)

        n_confs = molobj.GetNumConformers()
        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

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

            if self.show_progress:
                pbar.update(1)

        if self.show_progress:
            pbar.close()

        return properties_list


def get_properties_from_axyzc(
    atoms_str,
    coordinates,
    charge,
    spin,
    options=None,
    scr=constants.SCR,
    clean_tempfiles=False,
    cmd=G16_CMD,
    filename=G16_FILENAME,
    footer=None,
    **kwargs,
):

    if isinstance(scr, str):
        scr = pathlib.Path(scr)

    if not filename.endswith(".com"):
        filename += ".com"

    tempdir = tempfile.TemporaryDirectory(dir=scr, prefix="g16_")
    scr = pathlib.Path(tempdir.name)

    # write input file
    input_header = get_header(options, **kwargs)

    inputstr = get_inputfile(
        atoms_str, coordinates, charge, spin, input_header, footer=footer, title="g16 input"
    )

    with open(scr / filename, "w") as f:
        f.write(inputstr)

    # Run subprocess cmd
    cmd = " ".join([cmd, filename])
    _logger.debug(cmd)

    lines = shell.stream(cmd, cwd=scr)
    lines = list(lines)

    # for line in lines:
    #    print(line.strip())
    # print("="*80 )
    # print()

    termination_pattern = "Normal termination of Gaussian"
    idx = linesio.get_rev_index(lines, termination_pattern, stoppattern="File lengths")
    if idx is None:
        _logger.critical("Abnormal termination of Gaussian")
        return None

    # Parse properties from Gaussian output
    properties = read_properties(lines, options)

    # Clean scr dir. TODO: Does this work??
    if clean_tempfiles:
        tempdir.cleanup()

    return properties


def get_inputfile(atom_strs, coordinates, charge, spin, header, footer=None, **kwargs):
    """ """

    inputstr = header + 2 * "\n"
    inputstr += f"  {kwargs['title']}" + 2 * "\n"

    inputstr += f"{charge}  {spin} \n"
    for atom_str, coord in zip(atom_strs, coordinates):
        inputstr += f"{atom_str}  " + " ".join([str(x) for x in coord]) + "\n"
    inputstr += "\n"  # magic line

    if footer is not None:
        inputstr += footer + "\n"

    return inputstr


def get_header(options, **kwargs):
    """ Write G16 header """

    header = f"%mem={kwargs.pop('memory')}gb\n"
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


def read_properties(lines, options):
    """ Extract values from output depending on calculation options """

    # Collect readers
    readers = []

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


def read_properties_sp(lines):
    """
    Read Mulliken charges
    """
    properties = dict()

    for line in lines:
        if "SCF Done:  E(" in line:
            scf_energy = float(line.split()[4]) * units.hartree_to_kcalmol
            properties[COLUMN_SCF_ENERGY] = scf_energy
            break

    properties.update(get_mulliken_charges(lines))

    return properties


def read_properties_opt(lines):
    """ """


def get_mulliken_charges(lines):
    """ Read Mulliken charges """
    keywords = ["Mulliken charges:", "Sum of Mulliken charges"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)
    mulliken_charges = [float(x.split()[-1]) for x in lines[start + 2 : stop]]
    return {COLUMN_MULIKEN_CHARGES: mulliken_charges}


def get_hirsfeld_charges(lines):
    """ Read Hirsfeld charges - run a NBO calculation"""
    keywords = ["Hirshfeld charges,", "Hirshfeld charges with"]
    start, stop = linesio.get_indices_patterns(lines, keywords)
    hirshfeld_charges = [float(line.split()[2]) for line in lines[start + 2 : stop - 1]]

    return {COLUMN_HIRSHFELD_CHARGES: hirshfeld_charges}


def get_cm5_charges(lines):
    """ Read CM5 charges - run a NBO calculation"""
    keywords = ["Hirshfeld charges,", "Hirshfeld charges with"]
    start, stop = linesio.get_indices_patterns(lines, keywords)
    cm5_charges = [float(line.split()[-1]) for line in lines[start + 2 : stop - 1]]

    return {COLUMN_CM5_CHARGES: cm5_charges}


def get_nbo_bond_orders(lines):
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
    bond_idx_blocks = []
    block_idx = 0
    for line in lines[start + 1 : stop]:
        line = line.strip().split()
        if line:  # filter empty strings
            if "Atom" in line:
                continue

            if set("".join(line)).issubset(set("-")):
                bond_idx_blocks.append([])
                block_idx += 1
                continue

            bond_idx_blocks[block_idx - 1].append([float(x) for x in line[2:]])

    # Format matrix
    bond_order_matrix = bond_idx_blocks[0]
    for bond_idx_block in bond_idx_blocks[1:]:
        for i, bond_idx in enumerate(bond_idx_block):
            bond_order_matrix[i].extend(bond_idx)

    return {COLUMN_NBO_BONDORDER: bond_order_matrix}


def get_nmr_shielding_constants(lines):
    """ Read GIAO NMR shielding constants """
    keywords = ["Magnetic shielding tensor (ppm):", "************"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)

    shielding_constants = []
    for line in lines[start:stop]:
        if "Isotropic" in line:
            shielding_constants.append(float(line.strip().split()[4]))

    return {COLUMN_SHIELDING_CONSTANTS: shielding_constants}
