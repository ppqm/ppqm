"""
ORCA wrapper functions
"""

import functools
import logging
import pathlib
from collections import ChainMap
from typing import List

import numpy as np
from tqdm import tqdm

from ppqm import chembridge, constants, env, linesio, misc, shell, units
from ppqm.calculator import BaseCalculator
from ppqm.utils.files import WorkDir

ORCA_CMD = "orca"
ORCA_FILENAME = "_tmp_orca_input.inp"

COLUMN_COORD = "coord"
COLUMN_SCF_CONVERGED = "scf_converged"
COLUMN_SCF_ENERGY = "scf_energy"
COLUMN_GIBBS_FREE_ENERGY = "gibbs_free_energy"
COLUMN_ENTHALPY = "enthalpy"
COLUMN_ENTROPY = "entropy"
COLUMN_MULIKEN_CHARGES = "mulliken_charges"
COLUMN_LOEWDIN_CHARGES = "loewdin_charges"
COLUMN_HIRSHFELD_CHARGES = "hirshfeld_charges"
COLUMN_NBO_BONDORDER = "bond_orders"
COLUMN_STATIONARY_POINTS = "stationary_points"
COLUMN_VIBRATIONAL_FREQUENCIES = "vibrational_frequencies"
COLUMN_SHIELDING_CONSTANTS = "shielding_constants"

_logger = logging.getLogger("orca")


class OrcaCalculator(BaseCalculator):
    """Implementation of an Orca wrapper for ppqm."""

    def __init__(
        self,
        cmd=ORCA_CMD,
        filename=ORCA_FILENAME,
        show_progress=False,
        n_cores=1,
        memory=2,  # memory per core
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
            "memory": self.memory,  # memory per core
        }

        # Default Orca options
        self.options = {}

        self.health_check()

    def __repr__(self) -> str:
        return f"OrcaCalc(cmd={self.cmd}, scr={self.scr}, n_cores={self.n_cores}, memory={self.memory}gb)"

    def health_check(self):
        assert env.which(self.cmd), f"Cannot find {self.cmd}"

        # There is no such thing as "orca --version": https://orcaforum.kofo.mpg.de/viewtopic.php?f=8&t=8181
        stdout, _ = shell.execute(f'{self.cmd} idonotexist.inp | grep "Program"')

        try:
            stdout = stdout.split("\n")
            stdout = [x.strip() for x in stdout if "Program Version" in x]
            version = stdout[0].split(" ")[2]
            version = version.split(".")
            major, minor, patch = version
            self.VERSION = version
        except Exception:
            assert False, "unsupported Orca version"

        # If health check has gone through, update absolute path
        self.cmd = env.which(self.cmd)
        self.orca_options["cmd"] = self.cmd

    def calculate(self, molobj, options):
        """ """

        if self.n_cores and self.n_cores > 1:
            results = self.calculate_parallel(molobj, options)
        else:
            results = self.calculate_serial(molobj, options)

        return results

    def calculate_serial(self, molobj, options):
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
                **self.orca_options,
            )

            properties_list.append(properties)

            if self.show_progress:
                pbar.update(1)

        if self.show_progress:
            pbar.close()

        return properties_list

    def calculate_parallel(self, molobj, options, n_cores=None):

        _logger.debug("start orca multiprocessing pool")

        if not n_cores:
            n_cores = self.n_cores

        # If not singlet "spin" is part of options
        if "spin" in options.keys():
            spin = str(options.pop("spin"))
        else:
            spin = str(1)

        options_prime = ChainMap(options, self.options)
        options_prime = dict(options_prime)

        n_conformers = molobj.GetNumConformers()
        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

        coordinates_list = [
            np.asarray(conformer.GetPositions()) for conformer in molobj.GetConformers()
        ]

        # self.n_cores: how many cores are available (for parallel jobs + conformers)

        # don't use more cores than you have and allocate
        # the same number for each conformer
        n_procs_per_conformer = n_cores // n_conformers
        if n_procs_per_conformer < 1:
            # use at least one core
            self.orca_options["n_cores"] = 1
        else:
            self.orca_options["n_cores"] = n_procs_per_conformer

        _logger.info(f"Using {self.orca_options['n_cores']} core(s) per conformer")
        _logger.info(
            f"{n_conformers} conformer(s) in total on {n_procs_per_conformer * n_conformers} cores"
        )

        results = []

        func = functools.partial(
            get_properties_from_acxyz,
            atoms,
            charge,
            spin,
            options=options_prime,
            **self.orca_options,
        )

        results = misc.func_parallel(
            func,
            coordinates_list,
            n_cores=min(n_cores, n_conformers),
            n_jobs=n_conformers,
            show_progress=self.show_progress,
            title="ORCA",
        )

        return results


def get_properties_from_acxyz(atoms, charge, spin, coordinates, **kwargs):
    """ get properties from atoms, charge and coordinates """
    return get_properties_from_axyzc(atoms, coordinates, charge, spin, **kwargs)


def get_properties_from_axyzc(
    atoms_str,
    coordinates,
    charge,
    spin,
    options=None,
    scr=constants.SCR,
    keep_files=False,
    cmd=ORCA_CMD,
    filename=ORCA_FILENAME,
    **kwargs,
):

    # make sure orca is called with its full path in case no OrcaCalculator
    # object has been created. Don't run into avoidable runtime errors.
    cmd = env.which(cmd)

    if isinstance(scr, str):
        scr = pathlib.Path(scr)

    if not filename.endswith(".inp"):
        filename += ".inp"

    workdir = WorkDir(dir=scr, prefix="orca_", keep=keep_files)
    scr = workdir.get_path()

    # write input file
    input_header = get_header(options, **kwargs)

    inputstr = get_inputfile(atoms_str, coordinates, charge, spin, input_header)

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
        errors = read_error(lines)
        _logger.error("Abnormal termination of Orca")
        for error in errors:
            _logger.error(f"orca: {error}")
        return None

    # Parse properties from Orca output
    properties = read_properties(lines, len(atoms_str), options)

    properties[COLUMN_SCF_CONVERGED] = True  # This is asserted some lines earlier
    properties[COLUMN_COORD] = coordinates

    return properties


def get_inputfile(atom_strs, coordinates, charge, spin, header, **kwargs):
    """ """

    inputstr = header + 2 * "\n"

    # charge, spin, and coordinate section
    inputstr += f"*xyz {charge} {spin} \n"
    for atom_str, coord in zip(atom_strs, coordinates):
        inputstr += (
            f"{atom_str}".ljust(5) + " ".join(["{:.8f}".format(x).rjust(15) for x in coord]) + "\n"
        )
    inputstr += "*\n"
    inputstr += "\n"  # magic line

    return inputstr


def get_header(options, **kwargs):
    """ Write Orca header """

    header = "# ORCA input generated by ppqm" + 2 * "\n"

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


def read_error(lines: List[str]):
    """ Read the error message from orca log. So far I've only seen it between two headrules of exclamation marks"""

    error_lines = "!!!!!!!"
    patterns = [error_lines, error_lines]
    hr2, hr1 = linesio.get_rev_indices_patterns(lines, patterns, maxiter=50)
    errors = []

    if hr1 is None or hr2 is None:
        return errors

    for line in lines[hr1 + 1 : hr2]:
        line = " ".join(line.split())
        errors.append(line.strip().rstrip())

    return errors


def read_properties(lines, atom_number, options):
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

    if "Freq" in options or "NumFreq" in options:
        readers.append(get_vibrational_frequencies)
        readers.append(get_gibbs_free_energy)
        readers.append(get_enthalpy)
        readers.append(get_entropy)

    # these are always calculated by orca
    # always reported unless the print level is reduced
    if "MiniPrint" not in options:
        readers.append(get_mulliken_charges)
        readers.append(get_loewdin_charges)

    # get properties
    properties = dict()
    for reader in readers:
        new_properties = reader(lines, atom_number)
        if isinstance(new_properties, dict):
            properties.update(new_properties)
        else:
            _logger.error(f"Parser failed to read properties for reader {reader.__name__}")

    if "Freq" in options or "NumFreq" in options:
        imaginary_frequencies = len(
            [i for i in properties[COLUMN_VIBRATIONAL_FREQUENCIES] if i < 0]
        )
        if imaginary_frequencies == 0:
            properties[COLUMN_STATIONARY_POINTS] = "local_minimum"
        elif imaginary_frequencies == 1:
            properties[COLUMN_STATIONARY_POINTS] = "transition_state"
        else:
            properties[COLUMN_STATIONARY_POINTS] = "higher_order"

    return properties


def read_properties_sp(lines, atom_number):
    """
    Read Singlepoint Energy
    """
    properties = dict()

    for line in lines:
        if "FINAL SINGLE POINT ENERGY" in line:
            scf_energy = float(line.split()[4]) * units.hartree_to_kcalmol
            properties[COLUMN_SCF_ENERGY] = scf_energy
            break

    return properties


def get_mulliken_charges(lines, atom_number):
    """ Read Mulliken charges """
    pattern = "MULLIKEN ATOMIC CHARGES"
    start, stop = linesio.get_indices_pattern(lines, pattern, atom_number, 2)
    if [x for x in (start, stop) if x is None]:  # check if start or stop are None
        return None
    mulliken_charges = [float(x.split()[-1]) for x in lines[start:stop]]
    return {COLUMN_MULIKEN_CHARGES: mulliken_charges}


def get_loewdin_charges(lines, atom_number):
    """ Read Loewdin charges """
    pattern = "LOEWDIN ATOMIC CHARGES"
    start, stop = linesio.get_indices_pattern(lines, pattern, atom_number, 2)
    if [x for x in (start, stop) if x is None]:  # check if start or stop are None
        return None
    loewdin_charges = [float(x.split()[-1]) for x in lines[start:stop]]
    return {COLUMN_LOEWDIN_CHARGES: loewdin_charges}


def get_hirshfeld_charges(lines, atom_number):
    """ Read Hirsfeld charges """
    pattern = "HIRSHFELD ANALYSIS"
    start, stop = linesio.get_indices_pattern(lines, pattern, atom_number, 7)
    if [x for x in (start, stop) if x is None]:  # check if start or stop are None
        return None
    hirshfeld_charges = [float(line.split()[2]) for line in lines[start:stop]]

    return {COLUMN_HIRSHFELD_CHARGES: hirshfeld_charges}


def get_nmr_shielding_constants(lines, atom_number):
    """ Read GIAO NMR shielding constants """
    pattern = "CHEMICAL SHIELDING SUMMARY (ppm)"
    start, stop = linesio.get_indices_pattern(lines, pattern, atom_number, 6)
    if [x for x in (start, stop) if x is None]:  # check if start or stop are None
        return None
    shielding_constants = [float(line.split()[2]) for line in lines[start:stop]]

    return {COLUMN_SHIELDING_CONSTANTS: shielding_constants}


def get_vibrational_frequencies(lines, atom_number):
    """ Read vibrational frequencies """
    pattern = "VIBRATIONAL FREQUENCIES"
    degrees_of_freedom = 3 * atom_number
    start, stop = linesio.get_indices_pattern(lines, pattern, degrees_of_freedom, 5)
    if [x for x in (start, stop) if x is None]:  # check if start or stop are None
        return None
    vibrational_frequencies = [float(line.split()[1]) for line in lines[start:stop]]

    return {COLUMN_VIBRATIONAL_FREQUENCIES: vibrational_frequencies}


def get_gibbs_free_energy(lines, atom_number):
    """ Read Gibbs free energy """
    gibbs_free_energy = None  # Return None by default
    for line in lines:
        if "Final Gibbs free energy" in line:
            gibbs_free_energy = float(line.split()[5]) * units.hartree_to_kcalmol
            break

    return {COLUMN_GIBBS_FREE_ENERGY: gibbs_free_energy}


def get_enthalpy(lines, atom_number):
    """ Read enthalpy """
    enthalpy = None  # Return None by default
    for line in lines:
        if "Total enthalpy" in line:
            enthalpy = float(line.split()[3]) * units.hartree_to_kcalmol
            break

    return {COLUMN_ENTHALPY: enthalpy}


def get_entropy(lines, atom_number):
    """ Read entropy """
    entropy = None  # Return None by default
    for line in lines:
        if "Final entropy term" in line:
            entropy = float(line.split()[4]) * units.hartree_to_kcalmol
            break

    return {COLUMN_ENTROPY: entropy}
