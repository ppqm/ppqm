"""
ORCA wrapper functions
"""

import logging
import pathlib
import tempfile
from collections import ChainMap

from tqdm import tqdm

from ppqm import chembridge, constants, env, linesio, shell, units
from ppqm.calculator import BaseCalculator

ORCA_CMD = "orca"
ORCA_FILENAME = "_tmp_orca_input.inp"
ORCA_FILES = ["_tmp_orca_input.gbw", "_tmp_orca_input.out", "_tmp_orca_input.prop"]

COLUMN_SCF_ENERGY = "scf_energy"
COLUMN_MULIKEN_CHARGES = "mulliken_charges"
COLUMN_LOEWDIN_CHARGES = "loewdin_charges"
COLUMN_HIRSHFELD_CHARGES = "hirshfeld_charges"
COLUMN_NBO_BONDORDER = "bond_orders"
COLUMN_SHIELDING_CONSTANTS = "shielding_constants"

_logger = logging.getLogger("orca")

class OrcaCalculator(BaseCalculator):
    """Orca wrapper

    This class should not be used directly, use a class appropriate for your
    quantum calculations (e.g. MopacCalculator or GamessCalculator) instead.
    """

    def __init__(
        self,
        cmd=ORCA_CMD,
        filename=ORCA_FILENAME,
        show_progress=False,
        n_cores=1,
        memory=2,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.cmd = cmd
        self.filename = filename
        self.n_cores = n_cores
        self.memory = memory
        self.show_progress = show_progress

        self.orca_options = {
            "cmd": self.cmd,
            "scr": self.scr,
            "filename": self.filename,
            "n_cores": self.n_cores,
            "memory": self.memory,
        }

        # Default Orca options
        self.options = {}


        self.health_check()

    def __repr__(self) -> str:
        return f"OrcaCalc(cmd={self.cmd}, scr={self.scr}, n_cores={self.n_cores}, memory={self.memory}gb)"

    def health_check(self):
        assert env.which(self.cmd), f"Cannot find {self.cmd}"

        # There is no such thing as "orca --version": https://orcaforum.kofo.mpg.de/viewtopic.php?f=8&t=8181
        stdout, _ = shell.execute(f"{self.cmd} idonotexist.inp | grep \"Program\"")

        try:
            stdout = stdout.split("\n")
            stdout = [x.strip() for x in stdout if "Program Version" in x]
            version = stdout[0].split(" ")[2]
            version = version.split(".")
            major, minor, patch = version
            self.VERSION = version
        except Exception:
            assert False, "unsupported Orca version"

        assert int(major) == 4, "unsupported Orca version"
        assert int(minor) == 2, "unsupported Orca version"


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
                desc="orca(1)",
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
                **self.orca_options,
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
    cmd=ORCA_CMD,
    filename=ORCA_FILENAME,
    footer=None,
    **kwargs,
):

    if isinstance(scr, str):
        scr = pathlib.Path(scr)

    if not filename.endswith(".inp"):
        filename += ".inp"

    tempdir = tempfile.TemporaryDirectory(dir=scr, prefix="orca_")
    scr = pathlib.Path(tempdir.name)

    # write input file
    input_header = get_header(options, **kwargs)

    inputstr = get_inputfile(
        atoms_str, coordinates, charge, spin, input_header, footer=footer
    )

    with open(scr / filename, "w") as f:
        f.write(inputstr)

    # Run subprocess cmd

    cmd = " ".join([cmd, filename])
    _logger.debug(cmd)

    lines = shell.stream(cmd, cwd=scr)
    lines = list(lines)

    termination_pattern = "****ORCA TERMINATED NORMALLY****" 
    idx = linesio.get_rev_index(lines, termination_pattern)
    if idx is None:
        _logger.critical("Abnormal termination of Orca")
        return None

    # Parse properties from Orca output
    properties = read_properties(lines, options)

    # Clean scr dir. TODO: Does this work??
    if clean_tempfiles:
        tempdir.cleanup()

    return properties

def get_inputfile(atom_strs, coordinates, charge, spin, header, footer=None, **kwargs):
    """ """

    inputstr = header + 2 * "\n"

    # charge, spin, and coordinate section
    inputstr += f"*xyz {charge} {spin} \n"
    for atom_str, coord in zip(atom_strs, coordinates):
        inputstr += f"{atom_str}".ljust(5) + " ".join(["{:.8f}".format(x).rjust(15) for x in coord]) + "\n"
    inputstr += "*\n"
    inputstr += "\n"  # magic line

    if footer is not None:
        inputstr += footer + "\n"

    return inputstr

def get_header(options, **kwargs):
    """ Write Orca header """

    header = "# ORCA 4.2 input generated by ppqm" + 2 * "\n"

    header += "# Number of cores\n"
    header += f"%pal nprocs {kwargs.pop('n_cores')} end\n"
    header += "# RAM per core\n"
    header += f"%maxcore {1024 * kwargs.pop('memory')}" + 2 * "\n"

    for key, value in options.items():
        if (value is None) or (not value):
            header += f"! {key} \n"
        else:
            header += f"! {key}({value}) \n"

    return header

def read_properties(lines, options):
    """ Extract values from output depending on calculation options """

    # Collect readers
    readers = []

    if "Opt" in options:
        raise NotImplementedError("not implemented opt properties parser")
        # reader = read_properties_opt
    else:
        readers.append(read_properties_sp)

    if "Hirshfeld" in options:
        readers.append(get_hirshfeld_charges)

    if "NMR" in options:
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
    Read Singlepoint Energy
    """
    properties = dict()

    for line in lines:
        if "FINAL SINGLE POINT ENERGY" in line:
            scf_energy = float(line.split()[4]) * units.hartree_to_kcalmol
            properties[COLUMN_SCF_ENERGY] = scf_energy
            break

    properties.update(get_mulliken_charges(lines))
    properties.update(get_loewdin_charges(lines))

    return properties

def get_mulliken_charges(lines):
    """ Read Mulliken charges """
    keywords = ["MULLIKEN ATOMIC CHARGES", "Sum of atomic charges"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)
    mulliken_charges = [float(x.split()[-1]) for x in lines[start + 2 : stop]]
    return {COLUMN_MULIKEN_CHARGES: mulliken_charges}

def get_loewdin_charges(lines):
    """ Read Loewdin charges """
    keywords = ["LOEWDIN ATOMIC CHARGES", "LOEWDIN REDUCED ORBITAL CHARGES"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)
    loewdin_charges = [float(x.split()[-1]) for x in lines[start + 2 : stop - 2]]
    return {COLUMN_LOEWDIN_CHARGES: loewdin_charges}

def get_hirshfeld_charges(lines):
    """ Read Hirsfeld charges """
    keywords = ["HIRSHFELD ANALYSIS", "TIMINGS"]
    start, stop = linesio.get_indices_patterns(lines, keywords)
    hirshfeld_charges = [float(line.split()[2]) for line in lines[start + 7 : stop - 4]]

    return {COLUMN_HIRSHFELD_CHARGES: hirshfeld_charges}

def get_nmr_shielding_constants(lines):
    """ Read GIAO NMR shielding constants """
    keywords = ["CHEMICAL SHIELDING SUMMARY (ppm)", "Timings for individual modules:"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)

    start, stop = linesio.get_indices_patterns(lines, keywords)
    shielding_constants = [float(line.split()[2]) for line in lines[start + 6 : stop - 3]]

    return {COLUMN_SHIELDING_CONSTANTS: shielding_constants}