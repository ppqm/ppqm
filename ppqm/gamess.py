
import numpy as np
from pathlib import Path

from . import chembridge
from . import linesio
from . import constants
from . import calculator

GAMESS_CMD = "rungms"
GAMESS_SCR = "~/scr/"
GAMESS_USERSCR = "~/scr/"
GAMESS_ATOMLINE = "{:2s}    {:2.1f}    {:f}     {:f}    {:f}"


class GamessCalculator(calculator.CalculatorSkeleton):

    def __init__(self, cmd=GAMESS_CMD, scr="./"):

        self.cmd = cmd
        self.scr = scr

        # Ensure scrdir
        # if None, use tmpdir?
        Path(scr).mkdir(parents=True, exist_ok=True)

        # Constants
        self.atomline = GAMESS_ATOMLINE
        self.filename = "_tmp_gamess.inp"

        return

    def optimize(
            self, molobj,
            return_copy=True,
            return_properties=False,
            read_params=False):

        header = """ $basis gbasis={method} $end\n"""
        """$contrl runtyp=optimize icharg={charge} $end\n"""
        """$statpt opttol=0.0005 nstep=300 projct=.F. $end"""

        properties = self.calculate(molobj, header)

        return properties


    def calculate(self, molobj, header):

        atoms, _, charge = chembridge.molobj_to_axyzc(molobj, atom_type="str")

        # Create input
        txt = []
        for i in range(n_confs):
            coord = chembridge.molobj_to_coordinates(molobj, idx=i)
            tx = get_input(atoms, coord, charge, title=f"{title}_Conf{i}", **input_options)
            txt.append(tx)

        txt = "".join(txt)



        return



    def __repr__(self):
        return "GamessCalc(cmd={self.cmd},scr={self.scr})"



def calculate(molobj, header, **kwargs):

    inpstr = molobj_to_gmsinp(molobj, header)

    stdout, stderr = run(inpstr, **kwargs)

    return stdout, stderr


def prepare_atoms(atoms, coordinates):

    lines = []
    line = "{:2s}    {:2.1f}    {:f}     {:f}    {:f}"

    for atom, coord in zip(atoms, coordinates):
        iat = chembrigde.int_atom(atom)
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


def prepare_mol(filename, header, add_hydrogens=True):
    """
    """

    atoms = []
    coordinates = []

    with open(filename, 'r') as f:
        molfmt = f.read()
        mol = Chem.MolFromMolBlock(molfmt)

    # get formal charge
    charge = Chem.GetFormalCharge(mol)

    # Get coordinates
    conf = mol.GetConformer(0)
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz = [pos.x, pos.y, pos.z]
        coordinates.append(xyz)
        atoms.append(atom.GetSymbol())

    # set charge
    header = header.format(charge)
    lines = prepare_atoms(atoms, coordinates)

    return header + lines


def molobj_to_gmsinp(mol, header, conf_idx=-1):
    """
    RDKit Mol object to GAMESS input file

    args:
        mol - rdkit molobj
        header - str of gamess header

    returns:
        str - GAMESS input file
    """

    coordinates = []
    atoms = []

    # get formal charge
    charge = Chem.GetFormalCharge(mol)

    # Get coordinates
    conf = mol.GetConformer(conf_idx)
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xyz = [pos.x, pos.y, pos.z]
        coordinates.append(xyz)
        atoms.append(atom.GetSymbol())

    header = header.format(charge)
    lines = prepare_atoms(atoms, coordinates)

    return header + lines


def run(inpstr,
    cmd=GAMESS_CMD,
    scr=constants.SCR,
    filename=None,
    autoclean=True,
    gamess_scr="~/scr",
    gamess_userscr="~/scr",
    debug=False):
    """
    """

    if filename is None:
        pid = os.getpid()
        pid = str(pid)
        filename = "_tmp_gamess_run_" + pid + ".inp"

    pwd = os.getcwd()
    os.chdir(scr)

    with open(filename, 'w') as f:
        f.write(inpstr)

    cmd = cmd + " " + filename

    if debug:
        print(cmd)

    proc = subprocess.Popen(cmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if debug:
        print(stdout)

    if debug:
        print(stderr)

    if autoclean:
        clean(scr, filename)
        clean(gamess_scr, filename)
        clean(gamess_userscr, filename)

    # TODO no
    os.chdir(pwd)

    return stdout, stderr


def clean(scr, filename, debug=False):

    search = os.path.join(scr, filename.replace(".inp", "*"))
    files = glob.glob(search)
    for f in files:
        if debug: print("rm", f)
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


def read_errors(lines):

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

    idx = linesio.get_rev_index(lines, "FAILURE TO LOCATE STATIONARY POINT, TOO MANY STEPS TAKEN", stoppattern=safeword)
    if idx is not None:
        msg["error"] = "Failed to optimize molecule, too many steps taken. <br /> Try to displace atoms and re-calculate."
        return msg

    idx = linesio.get_rev_index(lines, "FAILURE TO LOCATE STATIONARY POINT, SCF HAS NOT CONVERGED", stoppattern=safeword)
    if idx is not None:
        msg["error"] = "Failed to optimize molecule, electrons too complicated. <br /> Try to displace atoms and re-calculate."
        return msg

    return msg


def read_properties(output):

    return


def read_properties_coordinates(output):

    properties = {}

    lines = output.split("\n")

    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    line = lines[idx]
    line = line.split("=")
    n_atoms = int(line[-1])

    idx = linesio.get_rev_index(lines, "FAILURE TO LOCATE STATIONARY POINT, TOO MANY STEPS TAKEN", stoppattern="NSEARCH")
    if idx is not None:
        properties["error"] = "Failed to optimize molecule, too many steps taken. <br /> Try to displace atoms and re-calculate."
        return properties

    idx = linesio.get_rev_index(lines, "FAILURE TO LOCATE STATIONARY POINT, SCF HAS NOT CONVERGED", stoppattern="NESERCH")
    if idx is not None:
        properties["error"] = "Failed to optimize molecule, electrons too complicated. <br /> Try to displace atoms and re-calculate."
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
    hof = float(line[4]) # kcal/mol

    properties[constants.COLUMN_ATOMS] = atoms
    properties[constants.COLUMN_COORDINATES] = coordinates
    properties["h"] = hof

    return properties


def read_properties_vibration(output):

    properties = {}

    lines = output.split("\n")

    # Get number of atoms
    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    line = lines[idx]
    line = line.split("=")
    n_atoms = int(line[-1])

    # Get heat of formation
    idx = linesio.get_rev_index(lines, "HEAT OF FORMATION IS")
    line = lines[idx]
    line = line.split()
    hof = float(line[4]) # kcal/mol

    # Check linear
    idx = linesio.get_index(lines, "THIS MOLECULE IS RECOGNIZED AS BEING LINEAR")
    is_linear = (idx is not None)

    # thermodynamic
    idx = linesio.get_rev_index(lines, "KJ/MOL    KJ/MOL    KJ/MOL   J/MOL-K")
    idx += 1
    values = np.zeros((5,6))
    for i in range(5):
        line = lines[idx +i]
        line = line.split()
        line = line[1:]
        line = [float(x) for x in line]
        values[i,:] = line

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
    properties["thermo"] = values
    properties["h"] = hof

    return properties


def read_properties_orbitals(output):

    properties = {}

    lines = output.split("\n")
    n_lines = len(lines)

    # Get number of atoms
    idx = linesio.get_index(lines, "TOTAL NUMBER OF ATOMS")
    line = lines[idx]
    line = line.split("=")
    n_atoms = int(line[-1])

    idx_start = linesio.get_index(lines, "EIGENVECTORS")
    idx_start += 4
    idx_end = linesio.get_index(lines, "END OF RHF CALCULATION", offset=idx_start)
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
    properties["stdout"] = output

    return properties


def read_properties_solvation(output):

    properties = {}

    lines = output.split("\n")
    n_lines = len(lines)

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

    total_non_polar = pierotti_cavitation_energy + dispersion_free_energy + repulsion_free_energy


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
