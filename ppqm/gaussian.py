import os
import logging
import pathlib
import tempfile
from collections import ChainMap

from tqdm import tqdm

from ppqm import chembridge, constants, shell, linesio, units
from ppqm.calculator import BaseCalculator


G16_CMD = "g16"
G16_FILENAME = "_tmp_g16_input.com"

_logger = logging.getLogger("g16")


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

        self.cmd = os.environ[cmd]
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
        pass

    def calculate(self, molobj, options):
        """ """

        if self.n_cores > 1:
            raise NotImplementedError("Parallel not implemented yet.")
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
                desc="g16(1)",
                **constants.TQDM_OPTIONS,
            )

        properties_list = []
        for conf_idx in range(n_confs):

            coord = chembridge.get_coordinates(molobj, confid=conf_idx)

            properties = get_properties_from_axyzc(
                atoms, coord, charge, spin, options=options_prime, **self.g16_options
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
        atoms_str, coordinates, charge, spin, input_header, title="g16 input"
    )

    with open(scr / filename, "w") as f:
        f.write(inputstr)

    # Run subprocess cmd
    cmd = " ".join([cmd, filename])
    _logger.debug(cmd)

    lines = shell.stream(cmd, cwd=scr)
    lines = list(lines)

    # for line in lines:
    #     print(line.strip())
    # print()
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


def get_inputfile(atom_strs, coordinates, charge, spin, header, **kwargs):
    """ """

    inputstr = header + 2 * "\n"
    inputstr += f"  {kwargs['title']}" + 2 * "\n"

    inputstr += f"{charge}  {spin} \n"
    for atom_str, coord in zip(atom_strs, coordinates):
        inputstr += f"{atom_str}  " + " ".join([str(x) for x in coord]) + "\n"
    inputstr += "\n"  # magic line

    return inputstr


def get_header(options, **kwargs):
    """ """

    header = f"%mem={kwargs.pop('memory')}gb\n"
    header += "# "
    for key, value in options.items():
        if value is None:
            header += f"{key} "

        else:
            header += f"{key}=("
            for subkey, subvalue in value.items():
                if subvalue is None:
                    header += f"{subkey}, "

                else:
                    header += f"{subkey}={subvalue}, "
            header += ") "

    return header


def read_properties(lines, options):
    """ """
    reader = None

    if "opt" in options:
        raise NotImplementedError("not implemented opt properties parser")
        # reader = read_properties_opt
    else:
        reader = read_properties_sp

    properties = reader(lines)

    return properties


def read_properties_sp(lines):
    """
    Read Mulliken charges,
    """

    properties = dict()

    for line in lines:
        if "SCF Done:  E(" in line:
            scf_energy = float(line.split()[4]) * units.hartree_to_kcalmol
            properties["total energy"] = scf_energy
            break

    properties["mulliken charges"] = get_mulliken_charges(lines)

    return properties


def read_properties_opt(lines):
    """ """
    pass


def get_mulliken_charges(lines):
    """ """

    keywords = ["Mulliken charges:", "Sum of Mulliken charges"]
    start, stop = linesio.get_rev_indices_patterns(lines, keywords)

    charges = [float(x.split()[-1]) for x in lines[start + 2 : stop]]

    return charges
