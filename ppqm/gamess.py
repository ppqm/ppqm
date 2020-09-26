
import numpy as np
import os
import glob
from collections import ChainMap

import rmsd

from .calculator import BaseCalculator
from . import chembridge
from . import linesio
from . import constants
from . import shell
from . import env

GAMESS_CMD = "rungms"
GAMESS_SCR = "~/scr/"
GAMESS_USERSCR = "~/scr/"
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


class GamessCalculator(BaseCalculator):

    def __init__(
        self,
        cmd=GAMESS_CMD,
        gamess_scr=GAMESS_SCR,
        gamess_userscr=GAMESS_USERSCR,
        filename=GAMESS_FILENAME,
        method_options=GAMESS_DEFAULT_OPTIONS,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cmd = cmd
        self.filename = filename

        self.method = method_options["method"]

        if "solvent" in method_options:
            self.solvent = method_options["solvent"]
        else:
            self.solvent = None

        self.gamess_options = {
            "cmd": self.cmd,
            "scr": self.scr,
            "gamess_scr": gamess_scr,
            "gamess_userscr": gamess_userscr
        }

        self._health_check()

        # Set calculation
        self._init_options()

    def _health_check(self):
        assert env.command_exists(self.cmd), (
            f"{self.cmd} was not found")

    def _init_options(self):

        self.options = dict()
        self.options["basis"] = {
            "gbasis": f"{self.method}",
        }
        self.options["contrl"] = {
            "scftyp": "rhf",
        }

        if self.solvent is not None:
            self.options["pcm"] = {
                "solvnt": f"{self.solvent}",
                "mxts": 15000,
                "icav": 1,
                "idisp": 1
            }
            self.options["tescav"] = {
                "mthall": 4,
                "ntsall": 60
            }

        return

    def _generate_options(self, optimize=True, hessian=False, gradient=False):

        if optimize:
            calculation = "optimize"
        elif hessian:
            calculation = "hessian"
        else:
            calculation = "energy"

        options = dict()
        options["contrl"] = {
            "runtyp": f"{calculation}",
        }

        if optimize:
            options["statpt"] = {
                "opttol": 0.005,
                "nstep": 300,
                "projct": False
            }

        return options

    def calculate(
        self,
        molobj,
        options
    ):
        """ """

        # Merge options
        options_prime = ChainMap(options, self.options)
        options_prime = dict(options_prime)
        options_prime["contrl"]["icharg"] = GAMESS_KEYWORD_CHARGE

        properties_list = []
        n_confs = molobj.GetNumConformers()

        atoms, _, charge = chembridge.molobj_to_axyzc(molobj, atom_type="str")

        for conf_idx in range(n_confs):

            coord = chembridge.molobj_get_coordinates(molobj, idx=conf_idx)
            properties = properties_from_axyzc(
                atoms,
                coord,
                charge,
                options_prime,
                **self.gamess_options
            )

            properties_list.append(properties)

        return properties_list

    def __repr__(self):
        return "GamessCalc(cmd={self.cmd},scr={self.scr},met={self.method})"


def properties_from_axyzc(
    atoms,
    coords,
    charge,
    options,
    return_stdout=False,
    **kwargs
):
    """
    """

    # Prepare input
    header = get_header(options)

    # set charge
    header = header.format(charge=charge)

    inptxt = get_input(atoms, coords, header)

    # Call GAMESS
    stdout, stderr = run_gamess(inptxt, **kwargs)

    # TODO Check stderr

    stdout = stdout.split("\n")

    if return_stdout:
        return stdout

    properties = get_properties(stdout)

    return properties


def prepare_atoms(atoms, coordinates):

    lines = []
    line = "{:2s}    {:2.1f}    {:f}     {:f}    {:f}"

    for atom, coord in zip(atoms, coordinates):
        iat = chembridge.int_atom(atom)
        lines.append(line.format(atom, iat, *coord))

    lines = [" $data", "Title", "C1"] + lines + [" $end"]

    return "\n".join(lines)


def prepare_xyz(filename, charge, header):
    """
    """

    atoms, coordinates = rmsd.get_coordinates_xyz("test.xyz")

    lines = prepare_atoms(atoms, coordinates)
    header = header.format(charge)

    gmsin = header + lines

    return gmsin


def get_input(atoms, coords, header):

    lines = header

    if lines[-1] != "\n":
        lines += "\n"

    lines += prepare_atoms(atoms, coords)

    return lines


def get_header(options):
    sections = []
    for section_name in options:
        sections.append(
            get_section(
                section_name,
                options[section_name]
            )
        )
    txt = "\n".join(sections)
    return txt


def get_section(section_name, options):
    section = f" ${section_name} "

    for key, val in options.items():
        if isinstance(val, bool):
            val = ".T." if val else ".F."

        section += f"{key}={val} "

    section += "$end"
    return section


def run_gamess(
    input_text,
    cmd=GAMESS_CMD,
    scr=constants.SCR,
    filename=GAMESS_FILENAME,
    gamess_scr="~/scr",
    gamess_userscr="~/scr",
    post_clean=True,
    pre_clean=True,
    debug=False
):
    """
    """

    assert env.command_exists(cmd), f"Could not find {cmd} in your enviroment"

    if pre_clean:
        clean(gamess_scr, filename)
        clean(gamess_userscr, filename)

    full_filename = os.path.join(scr, filename)

    with open(full_filename, 'w') as f:
        f.write(input_text)

    command = [cmd, filename]
    command = " ".join(command)

    stdout, stderr = shell.execute(command, chdir=scr)

    if debug:
        print(stdout)

    if debug:
        print(stderr)

    if post_clean:
        clean(gamess_scr, filename)
        clean(gamess_userscr, filename)

    return stdout, stderr


def clean(scr, filename, debug=False):

    scr = os.path.expanduser(scr)

    search = os.path.join(scr, filename.replace("inp", "*"))
    files = glob.glob(search)

    for f in files:
        if debug:
            print("rm", f)
        os.remove(f)

    return


def check_output(output):

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

    return True, ""


def get_errors(lines):

    if type(lines) == str:
        lines = lines.split("\n")

    msg = {}

    safeword = "NSERCH"

    key = "CHECK YOUR INPUT CHARGE AND MULTIPLICITY"
    idx = linesio.get_rev_index(lines, key, stoppattern=safeword)
    if idx is not None:
        line = lines[idx+1:idx+2]
        line = [x.strip().lower().capitalize() for x in line]
        line = ". ".join(line)
        line = line.replace("icharg=", "").replace("mult=", "")
        msg["error"] = line + ". Only multiplicity 1 allowed."
        return msg

    idx = linesio.get_rev_index(
        lines,
        "FAILURE TO LOCATE STATIONARY POINT, TOO MANY STEPS TAKEN",
        stoppattern=safeword
    )

    if idx is not None:
        msg["error"] = "TOO_MANY_STEPS"
        return msg

    idx = linesio.get_rev_index(
        lines,
        "FAILURE TO LOCATE STATIONARY POINT, SCF HAS NOT CONVERGED",
        stoppattern=safeword
    )

    if idx is not None:
        msg["error"] = "SCF_UNCONVERGED"
        return msg

    return msg


def get_properties(lines):

    # TODO Better keywords
    # TODO Make a reader list?
    # TODO Move SQM Specific properties
    # TODO Solvation

    reader = None

    runtyp = read_type(lines)

    if runtyp is None:
        return None

    method = read_method(lines)
    is_solvation = read_solvation(lines)

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


def read_solvation(lines):

    keyword = "INPUT FOR PCM SOLVATION CALCULATION"
    stoppattern = "ELECTRON INTEGRALS"

    idx = linesio.get_index(lines, keyword, stoppattern=stoppattern)
    if idx is not None:
        return True

    return False


def read_type(lines):

    idx = linesio.get_index(lines, "CONTRL OPTIONS")

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


def read_method(lines):

    # TODO Add disperion reader

    idx = linesio.get_index(lines, "BASIS OPTIONS")

    if idx is None:
        return None

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


def get_properties_coordinates(lines):

    properties = {}

    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    line = lines[idx]
    line = line.split("=")
    n_atoms = int(line[-1])

    idx = linesio.get_rev_index(
        lines,
        "FAILURE TO LOCATE STATIONARY POINT, TOO MANY STEPS TAKEN",
        stoppattern="NSEARCH"
    )

    if idx is not None:
        properties["error"] = "TOO_MANY_STEPS"
        properties[constants.COLUMN_COORDINATES] = None
        properties[constants.COLUMN_ATOMS] = None
        return properties

    idx = linesio.get_rev_index(
        lines,
        "FAILURE TO LOCATE STATIONARY POINT, SCF HAS NOT CONVERGED",
        stoppattern="NESERCH"
    )

    if idx is not None:
        properties["error"] = "SCF_UNCONVERGED"
        properties[constants.COLUMN_COORDINATES] = None
        properties[constants.COLUMN_ATOMS] = None
        return properties

    idx = linesio.get_rev_index(lines, "EQUILIBRIUM GEOMETRY LOCATED")
    idx += 4

    coordinates = np.zeros((n_atoms, 3))
    atoms = np.zeros(n_atoms, dtype=int)

    for i in range(n_atoms):
        line = lines[idx + i]
        line = line.split()
        atom = line[1].replace(".0", "")
        atom = int(atom)
        x = line[2]
        y = line[3]
        z = line[4]

        atoms[i] = atom
        coordinates[i][0] = x
        coordinates[i][1] = y
        coordinates[i][2] = z

    idx = linesio.get_rev_index(lines, "HEAT OF FORMATION IS")
    line = lines[idx]
    line = line.split()

    # energy in kcal/mol
    hof = float(line[4])

    properties[constants.COLUMN_ATOMS] = atoms
    properties[constants.COLUMN_COORDINATES] = coordinates
    properties[constants.COLUMN_ENERGY] = hof

    return properties


def get_properties_vibration(lines):

    properties = {}

    # Get number of atoms
    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    line = lines[idx]
    line = line.split("=")

    # Get heat of formation
    idx = linesio.get_rev_index(lines, "HEAT OF FORMATION IS")
    line = lines[idx]
    line = line.split()

    # energy in kcal/mol
    hof = float(line[4])

    # Check linear
    idx = linesio.get_index(
        lines,
        "THIS MOLECULE IS RECOGNIZED AS BEING LINEAR"
    )

    is_linear = (idx is not None)

    # thermodynamic
    idx = linesio.get_rev_index(lines, "KJ/MOL    KJ/MOL    KJ/MOL   J/MOL-K")
    idx += 1
    values = np.zeros((5, 6))

    for i in range(5):
        line = lines[idx + i]
        line = line.split()
        line = line[1:]
        line = [float(x) for x in line]
        values[i, :] = line

    # Get Vibrations
    idx_start = linesio.get_rev_index(lines, "FREQ(CM**-1)")
    idx_end = linesio.get_rev_index(lines, "THERMOCHEMISTRY AT T=  298.15 K")
    idx_start += 1
    idx_end -= 2
    vibrations = []
    intensities = []
    for i in range(idx_start, idx_end):
        line = lines[i]
        line = line.split()
        freq = line[1]
        freq = float(line[1])
        inte = line[-1]
        inte = float(inte)
        vibrations.append(freq)
        intensities.append(inte)

    # Cut and save vibration string for jsmol
    # based on number of vibrations and number of atoms
    idx = linesio.get_rev_index(lines, " TAKEN AS ROTATIONS AND TRANSLATIONS.")
    vib_lines = "\n".join(lines[idx:idx_start])

    idx_end = linesio.get_index(lines, "ELECTRON INTEGRALS")
    head_lines = "\n".join(lines[18:idx_end])

    properties["jsmol"] = head_lines + vib_lines
    properties["linear"] = is_linear
    properties["freq"] = np.array(vibrations)
    properties["intens"] = np.array(intensities)
    properties[COLUMN_THERMO] = values
    properties[constants.COLUMN_ENERGY] = hof

    return properties


def get_properties_orbitals(lines):

    properties = {}

    # Get number of atoms
    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    line = lines[idx]
    line = line.split("=")

    idx_start = linesio.get_index(lines, "EIGENVECTORS")
    idx_start += 4
    idx_end = linesio.get_index(
        lines,
        "END OF RHF CALCULATION",
        offset=idx_start
    )

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
            line = [float(x) for x in line]
            energies += line

        j += 1

    properties["orbitals"] = np.array(energies)
    properties["stdout"] = "\n".join(lines)

    return properties


def get_properties_solvation(lines):

    properties = {}

    # Get number of atoms
    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    line = lines[idx]
    line = line.split("=")
    n_atoms = int(line[-1])

    # Get solvation data,charge of molecule, surface area, dipole

    idx = linesio.get_rev_index(lines, "ELECTROSTATIC INTERACTION")
    line = lines[idx]
    line = line.split()
    electrostatic_interaction = float(line[-2])

    line = lines[idx+1].split()
    pierotti_cavitation_energy = float(line[-2])

    line = lines[idx+2].split()
    dispersion_free_energy = float(line[-2])

    line = lines[idx+3].split()
    repulsion_free_energy = float(line[-2])

    line = lines[idx+4].split()
    total_interaction = float(line[-2])

    total_non_polar = pierotti_cavitation_energy \
        + dispersion_free_energy \
        + repulsion_free_energy

    idx = linesio.get_index(lines, "CHARGE OF MOLECULE")
    line = lines[idx]
    line = line.split("=")
    charge = int(line[-1])

    idx = linesio.get_rev_index(lines, "SURFACE AREA")
    line = lines[idx]
    line = line.split()
    surface_area = line[2]
    surface_area = surface_area.replace("(A**2)", "")
    surface_area = float(surface_area)

    idx = linesio.get_rev_index(lines, "DEBYE")
    line = lines[idx+1]
    line = line.split()
    line = [float(x) for x in line]
    dxyz = line[0:3]
    dtot = line[-1]

    idx = linesio.get_rev_index(lines, "MOPAC CHARGES")
    idx += 3
    partial_charges = np.zeros(n_atoms)
    for i in range(n_atoms):
        line = lines[idx+i]
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
