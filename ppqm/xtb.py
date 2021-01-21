"""

Usage: xtb [options] <geometry> [options]

<geometry> may be provided as valid TM coordinate file (*coord in Bohr) or
in xmol format (*xyz in Ångström).

Options:

-c,--chrg INT     specify molecular charge as INT,
                    overrides .CHRG file and xcontrol option
-u,--uhf INT      specify Nalpha-Nbeta as INT,
                    overrides .UHF file and xcontrol option

-a,--acc REAL     accuracy for SCC calculation,
lower is better (default = 1.0)

    --iteration INT number of iterations in SCC (default = 250)

    --cycles       number of cycles in ANCopt (default = automatic)

    --gfn INT      specify parametrisation of GFN-xTB (default = 2)

    --qmdff        use QMDFF for single point (needs solvent-file)

    --tm           use TURBOMOLE for single point (needs control-file)

    --orca         use ORCA for single point (writes ORCA input)

    --mopac        use MOPAC for single point (writes MOPAC input)

    --periodic     uses periodic boundary conditions (in developement)

    --etemp REAL   electronic temperature (default = 300K)

    --vparam FILE  Parameter file for vTB calculation

    --xparam FILE  Parameter file for xTB calculation (not used)

-g,--gbsa [SOLVENT [STATE]] generalized born (GB) model with
                    solvent accessable surface area (SASA) model

    --pop          requests printout of Mulliken population analysis

    --molden       requests printout of molden file

    --dipole      requests dipole printout

    --wbo          requests Wiberg bond order printout

    --lmo          requests localization of orbitals

    --fod          requests FOD calculation, adjusts
                    electronic temperature to 12500 K if possible

    --scc, --sp    performs a single point calculation

    --vip          performs calculation of ionisation potential

    --vea          performs calculation of electron affinity

    --vipea        performs calculation of IP and EA

    --vomega       performs calculation of electrophilicity index

    --vfukui       calculate Fukui indicies using GFN-xTB

    --esp          calculate electrostatic potential on VdW-grid

    --stm          calculate STM image

    --grad         performs a gradient calculation

-o,--opt [LEVEL]  call ancopt(3) to perform a geometry optimization,
                    levels from crude, sloppy, loose, normal (default),
                    tight, verytight to extreme can be chosen

    --optts [LEVEL] [ROOT] call ancopt(3) to perform a transition state
                optimization, may need to perform a hessian calculation first

    --hess         perform a numerical hessian calculation on input geometry

    --ohess [LEVEL] perform a numerical hessian calculation on
                    an ancopt(3) optimized geometry

    --md           molecular dynamics simulation on start geometry

    --omd          molecular dynamics simulation on ancopt(3) optimized
                    geometry, a loose optimization level will be chosen.

    --metadyn [INT] meta dynamics simulation on start geometry
                    saving INT snapshots to bias the simulation

    --siman        conformational search by simulated annealing based on
                    molecular dynamics. Conformers are optimized with ancopt.

    --modef INT    modefollowing algorithm.  INT specifies the mode
                    that should be used for the modefollowing.

-I,--input FILE   use FILE as input source for xcontrol(7) instructions

    --namespace STRING give this xtb(1) run a namespace.
                    All files, even temporary ones, will be named accordingly
                    (might not work everywhere).

    --[no]copy     copies the xcontrol file at startup (default = true)

    --[no]restart  restarts calculation from xtbrestart (default = true)

-P,--parallel INT number of parallel processes

    --define       performs automatic check of input and terminate

    --version      print version and terminate

    --citation     print citation and terminate

    --license      print license and terminate

-v,--verbose      be more verbose (not supported in every unit)

-h,--help         show this message

Useful Maschine Settings:

export MKL_NUM_THREADS=<NCORE>
export OMP_THREAD_LIMIT=<NCORE>
export OMP_NUM_THREADS=${OMP_THREAD_LIMIT},1
export OMP_STACKSIZE=500m
ulimit -s unlimited

Output Conventions:

total energies are given in atomic units (Eh)
gaps/HL energies are given in eV

More information can be obtained by `man 1 xtb` and `man 7 xcontrol`
or at https://xtb-docs.readthedocs.io/en/latest/contents.html

"""

import copy
import functools
import logging
import multiprocessing
import os
import pathlib
import tempfile
from collections import ChainMap

import numpy as np
import rmsd
from tqdm import tqdm

from ppqm import chembridge, constants, env, linesio, misc, shell, units
from ppqm.calculator import BaseCalculator

XTB_CMD = "xtb"
XTB_FILENAME = "_tmp_xtb_input.xyz"
XTB_FILES = ["charges", "wbo", "xtbrestart", "xtbopt.coord", "xtbopt.log"]

COLUMN_ENERGY = "total_energy"
COLUMN_COORD = "coord"
COLUMN_ATOMS = "atoms"
COLUMN_GSOLV = "gsolv"
COLUMN_DIPOLE = "dipole"
COLUMN_CONVERGED = "is_converged"
COLUMN_STEPS = "opt_steps"

_logger = logging.getLogger("xtb")


class XtbCalculator(BaseCalculator):
    """

    TODO Add options documentation for XTB calculations

    """

    def __init__(
        self,
        cmd=XTB_CMD,
        filename=XTB_FILENAME,
        show_progress=True,
        n_cores=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.cmd = cmd
        self.filename = filename
        self.n_cores = n_cores
        self.show_progress = show_progress

        self.xtb_options = {
            "cmd": self.cmd,
            "scr": self.scr,
            "filename": self.filename,
        }

        # Default xtb options
        self.options = {}

    def _generate_options(self, optimize=True, hessian=False, gradient=False):
        # TODO
        options = ...
        return options

    def calculate(self, molobj, options):
        """ """

        if self.n_cores and self.n_cores > 1:
            results = self.calculate_parallel(molobj, options)

        else:
            results = self.calculate_serial(molobj, options)

        return results

    def calculate_serial(self, molobj, options):
        """ """

        # Merge options
        options_prime = ChainMap(options, self.options)
        options_prime = dict(options_prime)

        properties_list = []
        n_confs = molobj.GetNumConformers()

        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

        for conf_idx in range(n_confs):

            coord = chembridge.get_coordinates(molobj, confid=conf_idx)

            properties = get_properties_from_axyzc(
                atoms, coord, charge, options_prime, **self.xtb_options
            )

            properties_list.append(properties)

        return properties_list

    def calculate_parallel(self, molobj, options, n_cores=None):

        _logger.debug("start xtb multiprocessing pool")

        if not n_cores:
            n_cores = self.n_cores

        # Merge options
        options_prime = ChainMap(options, self.options)
        options_prime = dict(options_prime)

        atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)
        n_conformers = molobj.GetNumConformers()

        coordinates_list = [
            np.asarray(conformer.GetPositions()) for conformer in molobj.GetConformers()
        ]

        n_procs = min(n_cores, n_conformers)
        results = []

        func = functools.partial(
            get_properties_from_acxyz, atoms, charge, options=options_prime, **self.xtb_options
        )

        results = misc.func_parallel(
            func,
            coordinates_list,
            n_cores=n_procs,
            n_jobs=n_conformers,
            show_progress=self.show_progress,
            title="XTB",
        )

        return results

    def __repr__(self):
        return f"XtbCalc(cmd={self.cmd},scr={self.scr})"


def clean_dir(scr="./"):

    suffix = "/"
    if not scr.endswith(suffix):
        scr += suffix

    # TODO delete all tmp files made by xtb

    return


def health_check(config=None, cmd=XTB_CMD):
    """
    INCOMPLETE
    """

    path = env.which(cmd)

    assert path is not None

    return True


def get_properties_from_molobj(molobj, show_progress=True, **kwargs):
    """

    INCOMPLETE

    """

    n_conformers = molobj.GetNumConformers()

    if n_conformers == 0:
        raise ValueError("No conformers found in molecule")

    properties_list = []

    atoms, _, charge = chembridge.get_axyzc(molobj, confid=-1, atomfmt=str)

    if show_progress:
        pbar = tqdm(total=n_conformers, desc="XTB", **constants.TQDM_OPTIONS)

    # For conformers
    for conformer in molobj.GetConformers():
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)
        properties = get_properties_from_axyzc(atoms, coordinates, charge, **kwargs)
        properties_list.append(properties)

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    return properties_list


def get_properties_from_molobj_parallel(
    molobj, show_progress=True, n_cores=1, scr=None, options={}
):

    worker_kwargs = {"scr": scr, "n_cores": 1, "options": options}
    atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

    n_conformers = molobj.GetNumConformers()
    coordinates_list = [
        np.asarray(conformer.GetPositions()) for conformer in molobj.GetConformers()
    ]

    n_procs = min(n_cores, n_conformers)
    results = []

    func = functools.partial(get_properties_from_acxyz, atoms, charge, **worker_kwargs)

    results = misc.func_parallel(
        func,
        coordinates_list,
        n_cores=n_procs,
        show_progress=show_progress,
        title="XTB",
    )

    return results


def get_output_from_axyzc(
    atoms_str,
    coordinates,
    charge,
    options=None,
    scr=constants.SCR,
    use_tempfile=True,
    clean_tempfile=True,
    cmd=XTB_CMD,
    filename="_tmp_xtb_input.xyz",
    n_cores=1,
):
    """ NOT DONE """
    lines = ...
    return lines


def get_properties_from_acxyz(atoms, charge, coordinates, **kwargs):
    """ get properties from atoms, charge and coordinates """
    return get_properties_from_axyzc(atoms, coordinates, charge, **kwargs)


def get_properties_from_axyzc(
    atoms_str,
    coordinates,
    charge,
    options=None,
    scr=constants.SCR,
    use_tempfile=True,
    clean_tempfile=True,
    cmd=XTB_CMD,
    filename="_tmp_xtb_input.xyz",
    n_cores=1,
    **kwargs,
):
    """Get XTB properties from atoms, coordinates and charge for a molecule."""

    assert health_check(cmd=cmd)

    if isinstance(scr, str):
        scr = pathlib.Path(scr)

    if not filename.endswith(".xyz"):
        filename += ".xyz"

    if use_tempfile:
        temp = tempfile.TemporaryDirectory(dir=scr, prefix="xtb_")
        scr = pathlib.Path(temp.name)

    xtb_cmd = cmd

    # Write input file
    inputstr = rmsd.set_coordinates(atoms_str, coordinates, title="xtb input")

    with open(scr / filename, "w") as f:
        f.write(inputstr)

    # Set charge in file
    with open(scr / ".CHRG", "w") as f:
        f.write(str(charge))

    # Check and set threads
    n_omp = env.get_threads()
    if n_omp is not None and n_omp > 2:
        env.set_threads(n_cores)

    # Run subprocess command
    cmd = [cmd, f"{filename}"]

    if options is not None:
        cmd += parse_options(options)

    # Merge to string
    cmd = " ".join(cmd)
    cmd = f"cd {scr}; " + cmd

    _logger.debug(cmd)

    lines = shell.stream(cmd)
    lines = list(lines)

    # Reset threads
    if n_omp is not None:
        env.set_threads(n_omp)

    error_pattern = "abnormal termination of xtb"
    idx = linesio.get_rev_index(lines, error_pattern, stoppattern="#")
    if idx is not None:

        _logger.critical(error_pattern)

        idx = linesio.get_rev_index(lines, "ERROR")

        if idx is None:
            _logger.critical("could not read error message")

        else:

            for line in lines[idx + 1 : -2]:
                _logger.critical(line.strip())

        _logger.critical(cmd)
        _logger.critical("xtbexec " + env.which(xtb_cmd))
        _logger.critical("xtbpath " + os.environ.get("XTBPATH", ""))
        _logger.critical("xtbhome " + os.environ.get("XTBHOME", ""))

        return None

    # Parse properties from xtb output
    properties = read_properties(lines, options=options, scr=scr)

    # clean your room
    if use_tempfile and clean_tempfile:
        temp.cleanup()

    return properties


def calculate(molobj, confid=-1, show_progress=True, return_copy=True, **kwargs):
    """

    INCOMPLETE

    """

    # TODO Get coordinates
    atoms, coordinates, charge = chembridge.get_axyzc(molobj, confid=confid, atomfmt=int)

    properties = ...

    if return_copy:
        molobj = copy.deepcopy(molobj)

    n_conf = molobj.GetNumConformers()

    assert n_conf > 1, "No conformers to optimize"

    energies = np.zeros(n_conf)

    if show_progress:
        pbar = tqdm(total=n_conf, desc="XTB", **constants.TQDM_OPTIONS)

    # Iterat conformers and optimize with xtb
    for i in range(n_conf):

        atoms_str, coords, charge = chembridge.get_axyzc(molobj, confid=i, atomfmt=str)

        properties = get_properties_from_axyzc(atoms_str, coords, charge=charge, **kwargs)

        assert properties is not None, "Properties should never be None"

        coord = properties[COLUMN_COORD]
        chembridge.set_coordinates(molobj, coord, confid=i)

        if COLUMN_ENERGY in properties:
            total_energy = properties[COLUMN_ENERGY]
        else:
            total_energy = np.float("nan")

        energies[i] = total_energy

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    return properties


def optimize_molobj(molobj, return_copy=True, show_progress=True, **kwargs):
    """

    DEPRECIATED
    TODO Should move into useing calculate_

    TODO Embed energy into conformer?

    """

    if return_copy:
        molobj = copy.deepcopy(molobj)

    n_conf = molobj.GetNumConformers()

    assert n_conf > 1, "No conformers to optimize"

    energies = np.zeros(n_conf)

    if show_progress:
        pbar = tqdm(total=n_conf, desc="XTB", **constants.TQDM_OPTIONS)

    # Iterat conformers and optimize with xtb
    for i in range(n_conf):

        atoms_str, coords, charge = chembridge.get_axyzc(molobj, confid=i, atomfmt=str)

        properties = get_properties_from_axyzc(atoms_str, coords, charge=charge, **kwargs)

        assert properties is not None, "Properties should never be None"

        coord = properties[COLUMN_COORD]
        chembridge.set_coordinates(molobj, coord, confid=i)

        if COLUMN_ENERGY in properties:
            total_energy = properties[COLUMN_ENERGY]
        else:
            total_energy = np.float("nan")

        energies[i] = total_energy

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    return molobj, energies


def _worker_calculate_molobj(job, atoms=None, charge=None, **kwargs):
    """ INCOMPLETE """

    # Get job info
    i, coord = job

    # Get process information
    current = multiprocessing.current_process()
    pid = current.name
    pid = current._identity

    if len(pid) == 0:
        pid = 1
    else:
        pid = pid[0]

    pid = str(pid)
    scr = kwargs.get("scr")
    scr = os.path.join(scr, pid)
    kwargs["scr"] = scr

    pathlib.Path(scr).mkdir(parents=True, exist_ok=True)

    # Ensure only one thread per procs
    kwargs["n_cores"] = 1

    # TODO Should be general calculate with kwargs deciding to optimize

    properties = get_properties_from_axyzc(atoms, coord, charge=charge, **kwargs)

    return (i, properties)


def _worker_calculate_axyzc(job, **kwargs):
    """ INCOMPLETE """

    atoms, coord, charge = job

    # Get process information
    current = multiprocessing.current_process()
    pid = current.name
    pid = current._identity

    if len(pid) == 0:
        pid = 1
    else:
        pid = pid[0]

    pid = str(pid)
    pid = f"_calcxtb_{pid}"
    scr = kwargs.get("scr")
    scr = os.path.join(scr, pid)
    kwargs["scr"] = scr

    pathlib.Path(scr).mkdir(parents=True, exist_ok=True)

    # Ensure only one thread per procs
    kwargs["n_cores"] = 1

    properties = get_properties_from_axyzc(atoms, coord, charge, **kwargs)

    return properties


def procs_calculate_axyzc(molecules, n_cores=-1, show_progress=True, scr=None, cmd=XTB_CMD):
    """

    INCOMPLETE

    Start multiple subprocess over n_cores

    """
    results = None
    return results


def parallel_calculate_axyzc(
    molecules,
    options=None,
    n_cores=-1,
    show_progress=True,
    scr=None,
    cmd=XTB_CMD,
):
    """

    INCOMPLETE

    From lists of atoms, coords and charges. Return properties(dict) per
    molecule.

    :param molecules: List[Tuple[List[], array, int]]

    """

    if scr is None:
        scr = "_tmp_xtb_parallel_"

    if n_cores == -1:
        n_cores = env.get_available_cores()

    # Ensure scratch directories
    pathlib.Path(scr).mkdir(parents=True, exist_ok=True)

    if show_progress:
        pbar = tqdm(
            total=len(molecules),
            desc=f"XTB Parallel({n_cores})",
            **constants.TQDM_OPTIONS,
        )

    # Pool
    xtb_options = {"scr": scr, "cmd": cmd, "options": options}

    # TODO Add this worker test to test_xtb
    # TEST
    # properties = _worker_calculate_axyzc(
    #     molecules[0],
    #     debug=True,
    #     super_debug=True,
    #     **options
    # )
    # print(properties)
    # assert False

    func = functools.partial(_worker_calculate_axyzc, **xtb_options)
    p = multiprocessing.Pool(processes=n_cores)

    try:
        results_iter = p.imap(func, molecules, chunksize=1)
        results = []
        for result in results_iter:

            if COLUMN_ENERGY not in result:
                results[COLUMN_ENERGY] = np.float("nan")

            # Update the progress bar
            if show_progress:
                pbar.update(1)

            results.append(result)

    except KeyboardInterrupt:
        misc.eprint("got ^C while running pool of XTB workers...")
        p.terminate()

    except Exception as e:
        misc.eprint("got exception: %r, terminating the pool" % (e,))
        p.terminate()

    finally:
        p.terminate()

    # End the progress
    if show_progress:
        pbar.close()

    # TODO Clean scr dir for parallel folders, is the parallel folders needed
    # if we use tempfile?

    return results


def parallel_calculate_molobj(
    molobj,
    return_molobj=True,
    return_copy=True,
    return_energies=True,
    return_properties=False,
    update_coordinates=True,
    **kwargs,
):
    """
    INCOMPLETE
    """

    if return_copy:
        molobj = copy.deepcopy(molobj)

    num_conformers = molobj.GetNumConformers()
    assert num_conformers > 0, "No conformers to calculate"

    atoms, _, charge = chembridge.get_axyzc(molobj, atomfmt=str)

    jobs = []

    for i in range(num_conformers):
        conformer = molobj.GetConformer(id=i)
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)

        jobs.append((atoms, coordinates, charge))

    results = parallel_calculate_axyzc(jobs, **kwargs)

    if update_coordinates:
        for i, result in enumerate(results):

            coordinates = result.get(COLUMN_COORD, None)
            if coordinates is None:
                continue

            chembridge.set_coordinates(molobj, coordinates, confid=i)

    ret = tuple()

    if return_molobj:
        ret += (molobj,)

    if return_energies:
        energies = [result[COLUMN_ENERGY] for result in results]
        energies = np.array(energies)
        ret += (energies,)

    if return_properties:
        ret += (results,)

    return ret


# Readers


def read_status(lines):
    """"""
    keywords = [
        "Program stopped due to fatal error",
        "abnormal termination of xtb",
    ]
    stoppattern = "normal termination of xtb"

    idxs = linesio.get_rev_indices_patterns(lines, keywords, stoppattern=stoppattern)

    for idx in idxs:
        if idx is not None:
            return False

    return True


def parse_sum_table(lines):
    """ """

    properties = dict()

    for line in lines:

        if ":::" in line:
            continue

        if "..." in line:
            continue

        line = (
            line.replace("w/o", "without")
            .replace(":", "")
            .replace("->", "")
            .replace("/", "_")
            .strip()
        )
        line = line.split()

        if len(line) < 2:
            continue

        value = line[-2]
        value = float(value)
        # unit = line[-1]
        name = line[:-2]
        name = "_".join(name).lower()
        name = name.replace("-", "_").replace(".", "")

        properties[name] = float(value)

    return properties


def read_properties(lines, options=None, scr=None):
    """ Read output based on options or output """

    reader = None
    read_files = True

    if options is None:
        reader = read_properties_opt

    elif "vfukui" in options:
        reader = read_properties_fukui
        read_files = False

    elif "vomega" in options:
        reader = read_properties_omega
        read_files = False

    elif "opt" in options:
        reader = read_properties_opt

    else:
        reader = read_properties_sp

    properties = reader(lines)

    if scr is not None and read_files:
        # Parse file properties
        charges = get_charges(scr=scr)
        bonds, bondorders = get_wbo(scr=scr)

        properties["charges"] = charges
        properties["bonds"] = bonds
        properties["bondorders"] = bondorders

    return properties


def read_properties_sp(lines):
    """
    TODO read dipole moment
    TODO Inlcude units in docstring
    TODO GEOMETRY OPTIMIZATION CONVERGED AFTER 48 ITERATIONS

    electornic_energy is SCC energy

    """

    # TODO Better logging for crashed xtb
    if not read_status(lines):
        return None

    keywords = [
        "final structure:",
        "::                     SUMMARY                     ::",
        "Property Printout  ",
        "ITERATIONS",
    ]

    stoppattern = "CYCLE    "
    idxs = linesio.get_rev_indices_patterns(lines, keywords, stoppattern=stoppattern)
    idxs[0]
    idx_summary = idxs[1]
    idx_end_summary = idxs[2]
    idxs[3]

    if idx_summary is None:
        # TODO Better fix
        assert False, "uncaught xtb exception"

    # Get atom count
    keyword = "number of atoms"
    idx = linesio.get_index(lines, keyword)
    line = lines[idx]
    n_atoms = line.split()[-1]
    n_atoms = int(n_atoms)

    # Get energies
    idx_summary = idxs[1] + 1

    # :: total energy        +1
    # :: total w/o Gsasa/hb  +2
    # :: gradient norm       +3
    # :: HOMO-LUMO gap       +4
    # ::.....................+4
    # :: SCC energy          +5
    # :: -> isotropic ES     +6
    # :: -> anisotropic ES   +7
    # :: -> anisotropic XC   +8
    # :: -> dispersion       +9
    # :: -> Gsolv            +10
    # ::    -> Gborn         +11
    # ::    -> Gsasa         +12
    # ::    -> Ghb           +13
    # ::    -> Gshift        +14
    # :: repulsion energy    +15
    # :: add. restraining    +16

    prop_lines = lines[idx_summary : idx_end_summary - 2]
    prop_dict = parse_sum_table(prop_lines)

    # total_energy = prop_dict.get("total_energy", float("nan"))
    # gsolv = prop_dict.get("gsolv", float("nan"))
    # electronic_energy = prop_dict.get("scc_energy", float("nan"))

    properties = prop_dict

    # Get dipole
    dipole_str = "molecular dipole:"
    idx = linesio.get_rev_index(lines, dipole_str)
    if idx is None:
        dipole_tot = None
    else:
        idx += 3
        line = lines[idx]
        line = line.split()
        dipole_tot = line[-1]
        dipole_tot = float(dipole_tot)

    properties = {
        COLUMN_DIPOLE: dipole_tot,
        **properties,
    }

    # Get orbitals
    properties_orbitals = read_properties_orbitals(lines)
    properties = {**properties, **properties_orbitals}

    return properties


def read_properties_opt(lines, convert_coords=False, debug=False):
    """

    TODO read dipole moment
    TODO Inlcude units in docstring
    TODO GEOMETRY OPTIMIZATION CONVERGED AFTER 48 ITERATIONS

    electornic_energy is SCC energy

    """

    # TODO Better logging for crashed xtb
    if not read_status(lines):
        return None

    keywords = [
        "final structure:",
        "::                     SUMMARY                     ::",
        "Property Printout  ",
        "ITERATIONS",
    ]

    stoppattern = "CYCLE    "
    idxs = linesio.get_rev_indices_patterns(lines, keywords, stoppattern=stoppattern)
    idx_coord = idxs[0]
    idx_summary = idxs[1]
    idx_end_summary = idxs[2]
    idx_optimization = idxs[3]

    if idx_summary is None:
        assert False, "Uncaught xtb exception. Please submit issue with calculation"

    # Get atom count
    keyword = "number of atoms"
    idx = linesio.get_index(lines, keyword)
    line = lines[idx]
    n_atoms = line.split()[-1]
    n_atoms = int(n_atoms)

    # Get coordinates
    if idx_coord is None:
        coords = None
        atoms = None

    else:

        def parse_coordline(line):
            line = line.split()
            atom = line[0]
            coord = [float(x) for x in line[1:]]
            return atom, coord

        atoms = []
        coords = []
        for i in range(idx_coord + 4, idx_coord + 4 + n_atoms):
            line = lines[i]
            atom, coord = parse_coordline(line)
            atoms.append(atom)
            coords.append(coord)

        atoms = np.array(atoms)
        coords = np.array(coords)

        if convert_coords:
            coords *= units.bohr_to_aangstroem

    # Get energies
    idx_summary = idxs[1] + 1

    # :: total energy        +1
    # :: total w/o Gsasa/hb  +2
    # :: gradient norm       +3
    # :: HOMO-LUMO gap       +4
    # ::.....................+4
    # :: SCC energy          +5
    # :: -> isotropic ES     +6
    # :: -> anisotropic ES   +7
    # :: -> anisotropic XC   +8
    # :: -> dispersion       +9
    # :: -> Gsolv            +10
    # ::    -> Gborn         +11
    # ::    -> Gsasa         +12
    # ::    -> Ghb           +13
    # ::    -> Gshift        +14
    # :: repulsion energy    +15
    # :: add. restraining    +16

    prop_lines = lines[idx_summary : idx_end_summary - 2]
    prop_dict = parse_sum_table(prop_lines)

    # total_energy = prop_dict.get("total_energy", float("nan"))
    # gsolv = prop_dict.get("gsolv", float("nan"))
    # electronic_energy = prop_dict.get("scc_energy", float("nan"))

    properties = prop_dict

    # Get dipole
    dipole_str = "molecular dipole:"
    idx = linesio.get_rev_index(lines, dipole_str)
    if idx is None:
        dipole_tot = None
    else:
        idx += 3
        line = lines[idx]
        line = line.split()
        dipole_tot = line[-1]
        dipole_tot = float(dipole_tot)

    if idx_optimization is None:
        is_converged = None
        n_cycles = None

    else:

        line = lines[idx_optimization]
        if "FAILED" in line:
            is_converged = False
        else:
            is_converged = True

        line = line.split()
        n_cycles = line[-3]
        n_cycles = int(n_cycles)

    properties = {
        COLUMN_ATOMS: atoms,
        COLUMN_COORD: coords,
        COLUMN_DIPOLE: dipole_tot,
        COLUMN_CONVERGED: is_converged,
        COLUMN_STEPS: n_cycles,
        **properties,
    }

    return properties


def read_properties_omega(lines):
    """


    Format:

    Calculation of global electrophilicity index (IP+EA)²/(8·(IP-EA))
    Global electrophilicity index (eV):    0.0058
    """

    keywords = ["Global electrophilicity index"]
    indices = linesio.get_rev_indices_patterns(lines, keywords)

    if indices[0] is None:
        return None

    line = lines[indices[0]]
    line = line.split()
    global_index = line[-1]
    global_index = float(global_index)

    properties = {"global_electrophilicity_index": global_index}

    return properties


def read_properties_fukui(lines):
    """
    Read the Fukui properties fro XTB log

    format:
     #        f(+)     f(-)     f(0)
     1O      -0.086   -0.598   -0.342
     2H      -0.457   -0.201   -0.329
     3H      -0.457   -0.201   -0.329
    """

    keywords = ["Fukui index Calculation", "f(+)", "Property Printout"]

    indices = linesio.get_rev_indices_patterns(lines, keywords)

    if indices[0] is None:
        return None

    start_index = indices[1]
    end_index = indices[2]

    f_plus_list = []
    f_minus_list = []
    f_zero_list = []

    for i in range(start_index + 1, end_index - 1):
        line = lines[i]
        line = line.split()

        f_plus = float(line[1])
        f_minus = float(line[2])
        f_zero = float(line[3])

        f_plus_list.append(f_plus)
        f_minus_list.append(f_minus)
        f_zero_list.append(f_zero)

    f_plus_list = np.array(f_plus_list)
    f_minus_list = np.array(f_minus_list)
    f_zero_list = np.array(f_zero_list)

    properties = {
        "f_plus": f_plus_list,
        "f_minus": f_minus_list,
        "f_zero": f_zero_list,
    }

    return properties


def get_charges(scr=None):

    if scr is None:
        scr = pathlib.Path(".")

    filename = scr / "charges"

    if not filename.is_file():
        return None

    # read charges files from work dir
    charges = np.loadtxt(filename)

    return charges


def get_wbo(scr=None):

    if scr is None:
        scr = pathlib.Path(".")

    filename = scr / "wbo"

    if not filename.is_file():
        return None

    # Read WBO file
    with open(filename, "r") as f:
        lines = f.readlines()

    bonds, bondorders = read_wbo(lines)

    return bonds, bondorders


def read_wbo(lines):
    """"""
    # keyword = "Wiberg bond orders"

    bonds = []
    bondorders = []
    for line in lines:
        parts = line.strip().split()
        bondorders.append(float(parts[-1]))
        parts = parts[:2]
        parts = [int(x) - 1 for x in parts]
        parts = (min(parts), max(parts))
        bonds.append(parts)

    return bonds, bondorders


def read_properties_orbitals(lines, n_offset=2):
    """

    format:

         #    Occupation            Energy/Eh            Energy/eV
      -------------------------------------------------------------
       ...           ...                  ...                  ...
        62        2.0000           -0.3635471              -9.8926
        63        2.0000           -0.3540913              -9.6353 (HOMO)
        64                         -0.2808508              -7.6423 (LUMO)
       ...           ...                  ...                  ...

    """

    properties = dict()

    keywords = ["(HOMO)", "(LUMO)"]
    indices = linesio.get_rev_indices_patterns(lines, keywords)

    if indices[0] is None:
        return None

    idx_homo = indices[0]
    idx_lumo = indices[1]

    # check if this is the right place
    if idx_homo - idx_lumo != -1:
        return None

    # HOMO
    line = lines[idx_homo]
    line = line.split()
    energy_homo = float(line[2])

    properties["homo"] = energy_homo

    # HOMO Offsets
    for i in range(n_offset):
        line = lines[idx_homo - (i + 1)]
        line = line.strip().split()

        if len(line) < 3:
            continue

        value = line[2]
        properties[f"homo-{i+1}"] = float(value)

    # LUMO
    line = lines[idx_lumo]
    line = line.split()
    idx_lumo_col = 1
    energy_lumo = float(line[idx_lumo_col])

    properties["lumo"] = energy_lumo

    # Lumo Offsets
    for i in range(n_offset):
        line = lines[idx_lumo + (i + 1)]
        line = line.strip().split()

        if len(line) < 3:
            continue

        value = line[idx_lumo_col]
        properties[f"lumo+{i+1}"] = float(value)

    return properties


def parse_options(options, return_list=True):
    """ Parse dictionary/json of options, and return arg list for xtb """

    cmd_options = []

    for key, value in options.items():

        if value is not None:
            txt = f"--{key} {value}"
        else:
            txt = f"--{key}"

        cmd_options.append(txt)

    if return_list:
        return cmd_options

    cmd_options = " ".join(cmd_options)

    return cmd_options
