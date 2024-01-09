import copy
import gzip
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem, RDLogger  # type: ignore[import-untyped]
from rdkit.Chem import AllChem, Draw, rdFreeSASA, rdmolops  # type: ignore[import-untyped]
from rdkit.Chem.EnumerateStereoisomers import (  # type: ignore[import-untyped]
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore[import-untyped]
from rdkit.Chem.rdMolDescriptors import (  # type: ignore[import-untyped]
    CalcNumUnspecifiedAtomStereoCenters,
)

from ppqm import units

_logger = logging.getLogger(__name__)

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

# Get Van der Waals radii (angstrom)
PTABLE = Chem.GetPeriodicTable()

# spin-multiplicities 2,3,4,3,2 for the atoms H, C, N, O, F, respectively.
MULTIPLICITY = {}
MULTIPLICITY["H"] = 2
MULTIPLICITY["C"] = 3
MULTIPLICITY["N"] = 4
MULTIPLICITY["O"] = 3
MULTIPLICITY["F"] = 2
MULTIPLICITY["Cl"] = 2


ATOM_LIST = [
    x.strip()
    for x in [
        "h ",
        "he",
        "li",
        "be",
        "b ",
        "c ",
        "n ",
        "o ",
        "f ",
        "ne",
        "na",
        "mg",
        "al",
        "si",
        "p ",
        "s ",
        "cl",
        "ar",
        "k ",
        "ca",
        "sc",
        "ti",
        "v ",
        "cr",
        "mn",
        "fe",
        "co",
        "ni",
        "cu",
        "zn",
        "ga",
        "ge",
        "as",
        "se",
        "br",
        "kr",
        "rb",
        "sr",
        "y ",
        "zr",
        "nb",
        "mo",
        "tc",
        "ru",
        "rh",
        "pd",
        "ag",
        "cd",
        "in",
        "sn",
        "sb",
        "te",
        "i ",
        "xe",
        "cs",
        "ba",
        "la",
        "ce",
        "pr",
        "nd",
        "pm",
        "sm",
        "eu",
        "gd",
        "tb",
        "dy",
        "ho",
        "er",
        "tm",
        "yb",
        "lu",
        "hf",
        "ta",
        "w ",
        "re",
        "os",
        "ir",
        "pt",
        "au",
        "hg",
        "tl",
        "pb",
        "bi",
        "po",
        "at",
        "rn",
        "fr",
        "ra",
        "ac",
        "th",
        "pa",
        "u ",
        "np",
        "pu",
    ]
]


class Mol:
    """Meta class for typing rdkit functions"""

    def GetNumConformers(self) -> int:
        return 0


def axyzc_to_molobj(atoms: List[str], coord: np.ndarray, charge: int) -> Mol:
    """
    Get a molobj with one conformer, without any graph

    atoms - List(Str)
    coord - array
    charge - int

    """
    assert isinstance(atoms[0], str)

    n_atoms = len(atoms)
    atoms = [atom.capitalize() for atom in atoms]

    write_mol = Chem.RWMol()

    for i in range(n_atoms):
        a = Chem.Atom(atoms[i])
        write_mol.AddAtom(a)

    # Translate to mol
    mol: Mol = write_mol.GetMol()

    # Set coordinates / Conformer
    conformer = Chem.Conformer(n_atoms)
    conformer_set_coordinates(conformer, coord)
    mol.AddConformer(conformer, assignId=True)  # type: ignore

    # Set charge on a random atom, just not hydrogen
    rdatoms = list(mol.GetAtoms())  # type: ignore
    rdatom = None
    for rdatom in rdatoms:
        if rdatom.GetAtomicNum() != 1:
            break

    assert rdatom is not None

    rdatom.SetFormalCharge(charge)

    return mol


# def bonds_to_molobj(atoms, coord, charges, bonds, bondorders):
#     """
#     INCOMPLETE
#     """
#
#     # bonddict = {
#     #     1: Chem.BondType.SINGLE,
#     #     2: Chem.BondType.DOUBLE,
#     #     3: Chem.BondType.TRIPLE,
#     # }
#     #
#     # mw = Chem.RWMol()
#     #
#     # for atom in atoms:
#     #     mw.AddAtom(Chem.Atom(atom))
#     #
#     # for bond, order in zip(bonds, bondorders):
#     #     print(order, int(np.ceil(order)))
#     #     order = int(np.ceil(order))
#     #     if order < 1:
#     #         continue
#     #     order = bonddict[order]
#     #     mw.AddBond(*bond, order)
#     #
#     # for atom, charge in zip(mw.GetAtoms(), charges):
#     #     atom.SetFormalCharge(charge)
#     #
#     # Chem.SanitizeMol(mw)
#     # mw = Chem.RemoveHs(mw)
#     # NOT USED smiles = Chem.MolToSmiles(mw)
#
#     # return mw


def clean_sdf_header(sdfstr: str) -> str:

    sdfstr = str(sdfstr)
    for _ in range(2):
        i = sdfstr.index("\n")
        sdfstr = sdfstr[i + 1 :]
    sdfstr = "\n\n" + sdfstr

    return sdfstr


def conformer_set_coordinates(conformer: Chem.Conformer, coordinates: np.ndarray) -> None:  # type: ignore[no-any-unimported]
    for i, pos in enumerate(coordinates):
        conformer.SetAtomPosition(i, pos)


def copy_molobj(molobj: Mol) -> Mol:
    """Copy molobj graph, without conformers"""
    # The boolean signifies a fast copy, e.g. no conformers
    molobj = Chem.Mol(molobj, True)
    return molobj


def enumerate_stereocenters(
    molobj: Mol,
    max_num_unassigned: int = 3,
) -> Optional[List[Mol]]:
    """Find all un-assigned stereocenteres and assign them
    In case an error occurs, it is logged and the function returns None.
    """

    properties = get_properties_from_molobj(molobj)

    num_unassigned = CalcNumUnspecifiedAtomStereoCenters(molobj)
    if num_unassigned > max_num_unassigned:
        smi = Chem.MolToSmiles(molobj)
        lg.error("Molecule %s has too many unassigned stereocenters", smi)
        return None

    stereo_options = StereoEnumerationOptions(
        tryEmbedding=True,
        onlyUnassigned=True,
        maxIsomers=2**max_num_unassigned,
        unique=True,
        rand=1,
    )

    # NOTE try/except to capture
    # Pre-condition Violation\n\tStereo atoms should be specified before
    # specifying CIS/TRANS bond stereochemistry

    try:
        enumerator = EnumerateStereoisomers(molobj, options=stereo_options)
        isomers = list(enumerator)
    except RuntimeError:
        smi = Chem.MolToSmiles(molobj)
        lg.error("Stereo Enumeration for Molecule %s failed.", smi)
        return None

    # Set whatever properties the original molecule had
    for mol in isomers:
        set_properties_on_molobj(mol, properties)

    return isomers


def find_max_feature(smiles: str) -> str:
    """
    Split SMILES into compounds and return the compound with highest
    fingerprint density
    """

    smiles_list = smiles.split(".")
    n_smiles = len(smiles_list)

    if n_smiles < 2:
        return smiles

    dense_list = np.zeros(n_smiles, dtype=int)

    for i, smi in enumerate(smiles_list):

        molobj = smiles_to_molobj(smi)

        if molobj is None:
            dense_list[i] = 0
            continue

        fp = Chem.RDKFingerprint(molobj)
        dense_list[i] = fp.GetNumOnBits()

    # Select smiles with most bit features
    idx = np.argmax(dense_list)
    smiles = smiles_list[idx]

    return smiles


def find_max_str(smiles: str) -> str:
    """
    General functionality to choose a multi-smiles string, containing the
    longest string
    """
    smiles = max(smiles.split("."), key=len)
    return smiles


def get_atom_charges(molobj: Mol) -> np.ndarray:
    """Get atom charges from molobj"""
    atoms = molobj.GetAtoms()  # type: ignore[attr-defined]
    charges = [atom.GetFormalCharge() for atom in atoms]
    return np.array(charges)


def get_atom_int(atmstr: str) -> int:
    """Get atom number from atom label"""
    atom = atmstr.strip().lower()
    return ATOM_LIST.index(atom) + 1


def get_atom_str(iatm: int) -> str:
    """Get atom label from atom number"""
    atom = ATOM_LIST[iatm - 1]
    return atom.capitalize()


def get_atoms(mol: Mol, type: Callable = int) -> np.ndarray:
    """Get atoms from molecule in either int or str format"""

    rdatoms = mol.GetAtoms()  # type: ignore[attr-defined]
    rdatoms = list(rdatoms)

    atoms: Union[list, np.ndarray]

    if type == int:
        atoms = [a.GetAtomicNum() for a in rdatoms]

    elif type == str:
        atoms = [a.GetSymbol() for a in rdatoms]

    else:
        assert False, "Unknown type"

    atoms = np.array(atoms)

    return atoms


def get_axyzc(
    molobj: Mol, confid: int = -1, atomfmt: Callable = int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Get atoms, XYZ coordinates and formal charge of a molecule"""
    conformer = molobj.GetConformer(id=confid)  # type: ignore[attr-defined]
    coordinates = conformer.GetPositions()
    coordinates = np.array(coordinates)
    atoms = get_atoms(molobj, type=atomfmt)
    charge = get_charge(molobj)
    return atoms, coordinates, charge


def get_charge(molobj: Mol) -> int:
    charge: int = rdmolops.GetFormalCharge(molobj)
    return charge


def get_coordinates(molobj: Mol, confid: int = -1) -> np.ndarray:
    """ """
    confid = int(confid)  # rdkit needs int type
    conformer = molobj.GetConformer(id=confid)  # type: ignore[attr-defined]
    coordinates = np.array(conformer.GetPositions())
    return coordinates


def get_boltzmann_weights(
    energies: np.ndarray, temp: float = units.kelvin_room, k: float = units.k_kcalmolkelvin
) -> np.ndarray:
    """
    Calcualte boltzmann weights

    Assume energies in kcal/mol

    p_i =
        = \\frac{1}{Q}} {e^{- {\\varepsilon}_i / k T}
        = \\frac{e^{- {\varepsilon}_i / k T}}{\\sum_{j=1}^{M}{e^{- {\\varepsilon}_j / k T}}}

    """

    inv_kt = 1.0 / (k * temp)
    energies = copy.deepcopy(energies)
    energies -= energies.min()
    energies = np.exp(-energies * inv_kt)
    energy_sum = np.sum(energies)

    # translate energies to weights
    # weights = energies / energy_sum
    energies /= energy_sum

    return energies


def get_bonds(molobj: Mol) -> List[Tuple[int, int]]:
    """Get all bonds from molobj"""

    bonds = molobj.GetBonds()  # type: ignore[attr-defined]

    rtn = []

    for bond in bonds:

        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        # UNUSED t = bond.GetBondType()

        ar = min([a, b])
        br = max([a, b])

        rtn.append((ar, br))

    return rtn


def get_canonical_smiles(smiles: str) -> str:
    """Translate smiles into a canonical form"""
    molobj = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(molobj, canonical=True)
    return smiles


def get_center_of_mass(atoms: Union[List[int], np.ndarray], coordinates: np.ndarray) -> np.ndarray:
    """Calculate center of mass"""

    total_mass = np.sum(atoms)

    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    Z = coordinates[:, 2]

    R = np.zeros(3)

    R[0] = np.sum(atoms * X)
    R[1] = np.sum(atoms * Y)
    R[2] = np.sum(atoms * Z)
    R /= total_mass

    return R


def get_dipole_moments(molobj: Mol) -> np.ndarray:
    """
    Compute dipole moment for all conformers, using Gasteiger charges and
    coordinates from conformers

    Expects molobj to contain conformers

    """

    # Conformer independent
    AllChem.ComputeGasteigerCharges(molobj)

    atoms = molobj.GetAtoms()  # type: ignore[attr-defined]
    atoms_int = np.array([atom.GetAtomicNum() for atom in atoms])
    atoms_charge = np.array([atom.GetDoubleProp("_GasteigerCharge") for atom in atoms])

    # Calculate moments for each conformer
    moments = []
    for conformer in molobj.GetConformers():  # type: ignore[attr-defined]
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)

        total_moment = get_dipole_moment(atoms_int, coordinates, atoms_charge)
        moments.append(total_moment)

    return np.array(moments)


def get_dipole_moment(
    atoms: np.ndarray, coordinates: np.ndarray, charges: np.ndarray, is_centered: bool = False
) -> float:
    """

    from wikipedia:
    For a charged molecule the center of charge should be the reference point
    instead of the center of mass.

    """

    # total_charge = int(np.sum(charges))

    if not is_centered:
        center = get_center_of_mass(atoms, coordinates)
        coordinates = coordinates - center

    X = coordinates[:, 0] * charges
    Y = coordinates[:, 1] * charges
    Z = coordinates[:, 2] * charges

    x = np.sum(X)
    y = np.sum(Y)
    z = np.sum(Z)

    xyz = np.array([x, y, z])

    # Calculate total moment vector length
    total_moment: float = np.linalg.norm(xyz)  # type: ignore

    return total_moment


def get_inertia(atoms: Union[List[int], np.ndarray], coordinates: np.ndarray) -> np.ndarray:
    """Calculate inertia moments"""

    com = get_center_of_mass(atoms, coordinates)

    coordinates -= com

    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    Z = coordinates[:, 2]

    rxx = Y**2 + Z**2
    ryy = X**2 + Z**2
    rzz = X**2 + Y**2

    Ixx = atoms * rxx
    Iyy = atoms * ryy
    Izz = atoms * rzz

    Ixy = atoms * Y * X
    Ixz = atoms * X * Z
    Iyz = atoms * Y * Z

    Ixx_ = np.sum(Ixx)
    Iyy_ = np.sum(Iyy)
    Izz_ = np.sum(Izz)

    Ixy_ = np.sum(Ixy)
    Ixz_ = np.sum(Ixz)
    Iyz_ = np.sum(Iyz)

    inertia = np.zeros((3, 3))

    inertia[0, 0] = Ixx_
    inertia[1, 1] = Iyy_
    inertia[2, 2] = Izz_

    inertia[0, 1] = -Ixy_
    inertia[1, 0] = -Ixy_
    inertia[0, 2] = -Ixz_
    inertia[2, 0] = -Ixz_
    inertia[1, 2] = -Iyz_
    inertia[2, 1] = -Iyz_

    w, _ = np.linalg.eig(inertia)

    return w


def get_inertia_diag(atoms: Union[List[int], np.ndarray], coordinates: np.ndarray) -> np.ndarray:
    """Calculate the inertia diagonal vector"""

    com = get_center_of_mass(atoms, coordinates)

    coordinates -= com

    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    Z = coordinates[:, 2]

    rx2 = Y**2 + Z**2
    ry2 = X**2 + Z**2
    rz2 = X**2 + Y**2

    Ix = atoms * rx2
    Iy = atoms * ry2
    Iz = atoms * rz2

    Ix_ = np.sum(Ix)
    Iy_ = np.sum(Iy)
    Iz_ = np.sum(Iz)

    inertia = np.zeros(3)
    inertia[0] = Ix_
    inertia[1] = Iy_
    inertia[2] = Iz_

    return inertia


def get_inertia_ratio(inertia: np.ndarray) -> np.ndarray:
    """Sort intertia digonal and calculate the shape ratio"""

    inertia.sort()

    ratio = np.zeros(2)
    ratio[0] = inertia[0] / inertia[2]
    ratio[1] = inertia[1] / inertia[2]

    return ratio


def get_inertia_ratios(molobj: Mol) -> np.ndarray:
    """Get inertia ratios for all conformers"""

    atoms = molobj.GetAtoms()  # type: ignore[attr-defined]
    atoms_int = np.array([atom.GetAtomicNum() for atom in atoms])

    # Calculate moments for each conformer
    ratios = []
    for conformer in molobj.GetConformers():  # type: ignore[attr-defined]
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)

        inertia = get_inertia_diag(atoms_int, coordinates)
        ratio = get_inertia_ratio(inertia)
        ratios.append(ratio)

    return np.array(ratios)


def get_properties_from_molobj(molobj: Mol) -> dict:
    """Get properties from molobj"""
    properties: dict = molobj.GetPropsAsDict()  # type: ignore[attr-defined]
    return properties


def get_sasa(molobj: Mol, extra_radius: float = 0.0) -> np.ndarray:
    """Get solvent accessible surface area per atom

    :param molobj: Molecule with 3D conformers
    :param extra_radius: Constant addition to the atom radii's

    :return sasa: List of area, for each conformer

    """

    radii = [PTABLE.GetRvdw(atom.GetAtomicNum()) for atom in molobj.GetAtoms()]  # type: ignore[attr-defined]

    n = molobj.GetNumConformers()

    radii = [r + extra_radius for r in radii]
    sasas = np.zeros(n)

    for i in range(n):
        sasa = rdFreeSASA.CalcSASA(molobj, radii, confIdx=i)
        sasas[i] = sasa

    return sasas


def get_torsions(mol: Mol) -> np.ndarray:
    """
    Get indices of all torsion pairs All heavy atoms.
    One end can be Hydrogen.

    return
        indicies - array of four atom indicies for each torsional angle found
    """

    any_atom = "[*]"
    smarts = "~".join([any_atom, any_atom, any_atom, any_atom])

    atoms = get_atoms(mol, type=str)

    idxs = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))  # type: ignore
    idxs = [list(x) for x in idxs]
    idxs = np.array(idxs)

    rtnidxs = []

    for idxgroup in idxs:

        these_atoms = atoms[idxgroup]

        (idx_hydrogen,) = np.where(these_atoms == "H")
        n_hydrogen = len(idx_hydrogen)

        if n_hydrogen > 1:
            continue
        elif n_hydrogen > 0:
            if idx_hydrogen[0] == 1:
                continue
            if idx_hydrogen[0] == 2:
                continue

        rtnidxs.append(idxgroup)

    return np.array(rtnidxs, dtype=int)


def get_undefined_stereocenters(molobj: Mol) -> int:
    """Count number of undefined steorecenter in molobj"""
    chiral_centers = dict(Chem.FindMolChiralCenters(molobj, includeUnassigned=True))
    n_undefined_centers = sum(1 for (x, y) in chiral_centers.items() if y == "?")
    return n_undefined_centers


def molobj_add_conformer(molobj: Mol, coordinates: np.ndarray) -> None:
    """Append coordinates as a new conformer to molobj"""
    conf = Chem.Conformer(len(coordinates))
    for i, coordinate in enumerate(coordinates):
        conf.SetAtomPosition(i, coordinate)
    molobj.AddConformer(conf, assignId=True)  # type: ignore[attr-defined]


def molobj_check_distances(
    molobj: Mol, min_cutoff: Optional[float] = 0.001, max_cutoff: Optional[float] = 3.0
) -> np.ndarray:
    """
    For some atom_types in UFF, rdkit will fail optimization and stick multiple
    atoms ontop of eachother

    Known problems in CS(F3)

    Return array(len(conformer)) with
    0 - okay bond
    1 - problem bond

    """

    n_confs = molobj.GetNumConformers()

    status = []

    for i in range(n_confs):

        # TODO Get uppertriangular instead, with no diagonal
        dist = Chem.rdmolops.Get3DDistanceMatrix(molobj, confId=i)

        np.fill_diagonal(dist, 10.0)
        min_dist = np.min(dist)

        np.fill_diagonal(dist, 0.0)
        max_dist = np.max(dist)

        this = 0

        if min_dist and min_dist < min_cutoff:
            this = 1

        if max_dist and max_dist > max_cutoff:
            this = 1

        status.append(this)

    return np.array(status)


def molobj_select_conformers(molobj: Mol, idxs: List[int]) -> Mol:
    """
    Filter function. Return molobj only with conformers with index in idxs.

    :param molobj: Molecule with number of conformers
    :param idx: List of indices
    :return molobj_prime: Molecule with filtered conformers
    """

    molobj_prime = copy_molobj(molobj)

    for idx in idxs:

        # rdkit requires int
        idx = int(idx)

        conf = molobj.GetConformer(id=idx)  # type: ignore[attr-defined]
        molobj_prime.AddConformer(conf, assignId=True)  # type: ignore[attr-defined]

    return molobj_prime


def molobj_set_coordinates(molobj: Mol, coordinates: np.ndarray, confid: int = -1) -> None:
    conformer = molobj.GetConformer(id=confid)  # type: ignore[attr-defined]
    conformer_set_coordinates(conformer, coordinates)


def molobjs_to_molobj(molobjs: List[Mol]) -> Mol:
    """
    take list of molobjs and merge into molobj with conformers

    IMPORTANT: expects all molobjs to be same graph and same atom order!
    """

    molobj = copy_molobj(molobjs[0])
    n_molecules = len(molobjs)

    atoms = list(get_atoms(molobjs[0], type=int))

    for idx in range(n_molecules):

        # Test we don't mix and match molecules
        assert molobjs[idx].GetNumConformers() == 1
        atoms_prime = list(get_atoms(molobjs[idx]))
        assert atoms == atoms_prime, "Cannot merge two different molecules"

        conf = molobjs[idx].GetConformer(id=-1)  # type: ignore[attr-defined]
        molobj.AddConformer(conf, assignId=True)  # type: ignore[attr-defined]

    return molobj


def molobjs_to_properties(molobjs: List[Mol]) -> Dict[str, List[Any]]:
    """Return a dictionary of every property found in the molobj.

    :param molobjs: Iter[Mol] List of molobjs
    :return properties: Dict[Str, List[Value]]
    """

    all_properties = []
    keys = []

    for molobj in molobjs:
        properties = molobj.GetPropsAsDict()  # type: ignore[attr-defined]
        all_properties.append(properties)

        keys += list(properties.keys())

    keys = np.unique(keys)

    rtn_values: dict = {key: [] for key in keys}

    for properties in all_properties:
        for key in keys:

            if key in properties:
                value = properties[key]
            else:
                value = None

            rtn_values[key].append(value)

    return rtn_values


def molobj_to_mol2(molobj: Mol, charges: Optional[np.ndarray] = None) -> str:
    """
    https://www.mdanalysis.org/docs/_modules/MDAnalysis/coordinates/MOL2.html
    """

    # Bonds
    bond_lines = ["@<TRIPOS>BOND"]
    bond_fmt = "{0:>5} {1:>5} {2:>5} {3:>2}"
    bonds = list(molobj.GetBonds())  # type: ignore[attr-defined]
    n_bonds = len(bonds)
    for i, bond in enumerate(bonds):
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        t = bond.GetBondType()
        tf = bond.GetBondTypeAsDouble()

        if tf.is_integer():
            t = int(t)
        else:
            t = "ar"

        bond = bond_fmt.format(i + 1, a + 1, b + 1, t)
        bond_lines.append(bond)

    bond_lines.append("\n")
    bond_lines_ = "\n".join(bond_lines)

    # Atoms
    atom_lines = ["@<TRIPOS>ATOM"]
    atom_fmt = "{0:>4} {1:>4} {2:>13.4f} {3:>9.4f} {4:>9.4f} {5:>4} {6} {7} {8:>7.4f}"
    atoms = list(molobj.GetAtoms())  # type: ignore[attr-defined]
    # atoms_int = [atom.GetAtomicNum() for atom in atoms]
    # atoms_int = np.array(atoms_int)
    atoms_str = [atom.GetSymbol() for atom in atoms]
    n_atoms = len(atoms)
    conformer = molobj.GetConformer()  # type: ignore[attr-defined]
    coordinates = conformer.GetPositions()
    coordinates = np.array(coordinates)
    # np.unique(atoms_int)

    if charges is None:
        charges = np.zeros(n_atoms)

    atm_i = 1

    for j in range(n_atoms):

        name = atoms_str[j]
        pos0 = coordinates[j, 0]
        pos1 = coordinates[j, 1]
        pos2 = coordinates[j, 2]
        typ = atoms_str[j]
        resid = 0
        resname = "MOL"
        charge = charges[j]

        atmstr = atom_fmt.format(j + 1, name, pos0, pos1, pos2, typ, resid, resname, charge)
        atom_lines.append(atmstr)

        atm_i += 1
        continue

    atom_lines.append("")
    atom_lines_ = "\n".join(atom_lines)

    # Complete
    checksumstr = f"{n_atoms} {n_bonds} 0 0 0"
    head_lines = ["@<TRIPOS>MOLECULE", "TITLE"]
    head_lines += [checksumstr, "SMALL", "MULLIKEN_CHARGES", "NAME"]
    head_lines.append("")
    head_lines_ = "\n".join(head_lines)

    rtnstr = head_lines_ + atom_lines_ + bond_lines_

    return rtnstr


def molobj_to_molobjs(molobj: Mol) -> List[Mol]:
    """Expand a molobj conformer into a list of molobjs"""

    molobj_prime = copy_molobj(molobj)

    molobjs = []

    for _, conf in enumerate(molobj.GetConformers()):  # type: ignore[attr-defined]

        molobj_psi = copy.deepcopy(molobj_prime)
        molobj_psi.AddConformer(conf, assignId=True)  # type: ignore[attr-defined]
        molobjs.append(molobj_psi)

    return molobjs


def molobj_to_sdfstr(mol: Mol, use_v3000: bool = False, include_properties: bool = False) -> str:
    """Get SDF string from Mol"""

    n_confs = mol.GetNumConformers()

    if n_confs == 0:
        mol = copy.deepcopy(mol)
        AllChem.Compute2DCoords(mol)
        n_confs = 1

    txts = []

    if include_properties:

        sio = StringIO()
        w = Chem.SDWriter(sio)

        if use_v3000:
            w.SetForceV3000(1)

        for i in range(n_confs):
            w.write(mol, confId=i)
            w.flush()
            txt = sio.getvalue()
            txts.append(txt)

    else:

        for i in range(n_confs):
            txt = Chem.MolToMolBlock(mol, confId=i, forceV3000=use_v3000)
            txts += [txt]

    return "$$$$\n".join(txts)


def molobj_to_smiles(
    molobj: Mol,
    remove_hs: bool = True,
    sanitize: bool = True,
    canonical: bool = True,
    kekulize: bool = False,
    remove_stereo: bool = False,
) -> str:

    if remove_stereo:
        rdmolops.RemoveStereochemistry(molobj)

    if sanitize:
        Chem.SanitizeMol(molobj)

    if kekulize:
        Chem.Kekulize(molobj, clearAromaticFlags=True)

    if remove_hs:
        molobj = Chem.RemoveHs(molobj)

    smiles: str = Chem.MolToSmiles(molobj, canonical=canonical, kekuleSmiles=kekulize)

    return smiles


def molobj_to_svgstr(
    molobj: Mol,
    use_2d: bool = True,
    highlights: Optional[List[int]] = None,
    pretty: bool = False,
    removeHs: bool = False,
) -> str:
    """
    Returns SVG in string format
    """

    if removeHs:
        molobj = Chem.RemoveHs(molobj)

    if use_2d:
        molobj = copy_molobj(molobj)
        AllChem.Compute2DCoords(molobj)

    svg: str = Draw.MolsToGridImage(
        [molobj],
        molsPerRow=1,
        subImgSize=(400, 400),
        useSVG=True,
        highlightAtomLists=[highlights],
    )

    svg = svg.replace("xmlns:svg", "xmlns")

    if pretty:

        svg_ = svg.split("\n")

        for i, line in enumerate(svg):

            # Atom letters
            if "text" in line:

                replacetext = "font-size"
                borderline = "fill:none;fill-opacity:1;stroke:#FFFFFF;stroke-width:10px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;"

                # Add border to text
                border_text = line
                border_text = border_text.replace("stroke:none;", "")
                border_text = border_text.replace(replacetext, borderline + replacetext)

                svg_[i] = border_text + "\n" + line

                continue

            if "path" in line:

                # thicker lines
                line = line.replace("stroke-width:2px", "stroke-width:3px")
                svg_[i] = line

        svg = "\n".join(svg_)

    svg = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")

    return svg


def neutralize_molobj(molobj: Mol) -> Mol:
    """
    Get a neutral protonation state of molobj and remove explicit hydrogens
    """

    total_charge = rdmolops.GetFormalCharge(molobj)

    if total_charge == 0:
        return molobj

    molobj = rdMolStandardize.ChargeParent(molobj)
    return molobj


def read(filename: Path, remove_hs: bool = False, sanitize: bool = True) -> Iterator[Mol]:
    """
    General function to read files with different extensions and return molobjs

    .sdf
    .sdf.gz
    .smi
    .smi.gz
    .csv
    .csv.gz

    :return molobjs:
    """

    suffix = filename.suffix
    print(suffix)

    if suffix == ".sdf" or suffix == ".mol":

        suppl = Chem.SDMolSupplier(str(filename), removeHs=remove_hs, sanitize=sanitize)

    elif suffix == ".sdf.gz":

        fobj = gzip.open(filename)
        suppl = Chem.ForwardSDMolSupplier(fobj, removeHs=remove_hs, sanitize=sanitize)

    elif suffix == ".smi":

        f = open(filename, "r")
        suppl = read_smi(f)

    elif suffix == ".smi.gz":

        fobj = gzip.open(filename)
        suppl = read_smi(fobj)  # type: ignore

    else:
        raise ValueError(f"Could not read {filename}")

    return suppl  # type: ignore


def read_smi(f: Iterable[str]) -> Iterator[Mol]:
    """
    Read smiles and yield generated molobjs with 2D coords

    :param f: iterable, either list or file
    :yields molobjs: Iteration of molobjs
    """

    for line in f:
        line = line.strip()

        # if includes_name:
        #     line = line.split()
        #     assert len(line) == 2
        #     name = line[1]
        #     line = line[0]

        molobj: Optional[Mol] = smiles_to_molobj(line)

        if molobj is None:
            _logger.error(f"Unable to parse {line}")
            continue

        # Compute 2D coords by default, to keep stereo
        AllChem.Compute2DCoords(molobj)

        yield molobj

    return


def sdfstrs_to_molobjs(sdfs: str, remove_hs: bool = False) -> List[Mol]:
    """

    From a string of multiple SDF structures
    return a List of Mol objs
    """

    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdfs, removeHs=remove_hs)
    molobjs = [mol for mol in suppl]

    return molobjs


def sdfstr_to_molobj(sdfstr: str, remove_hs: bool = False) -> Optional[Mol]:
    """Convert SDF string to Mol"""

    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdfstr, removeHs=remove_hs)
    try:
        molobj: Optional[Mol] = next(suppl)
    except StopIteration:
        molobj = None
    return molobj


def sdfstr_to_smiles(sdfstr: str, remove_hs: bool = False) -> str:
    """SDF to SMILES converter"""
    mol = Chem.MolFromMolBlock(sdfstr, removeHs=remove_hs)
    smiles: str = Chem.MolToSmiles(mol)
    return smiles


def smiles_to_molobj(
    smiles: str, compute_2d: bool = False, add_hydrogens: bool = True
) -> Optional[Mol]:

    molobj: Mol = Chem.MolFromSmiles(smiles)

    if molobj is None:
        return None

    if add_hydrogens:
        molobj = rdmolops.AddHs(molobj, addCoords=True)

    if compute_2d:
        AllChem.Compute2DCoords(molobj)

    return molobj


def set_properties_on_molobj(molobj: Mol, properties: Dict) -> None:
    """Set dictionary of properties to a Mol obj"""
    for key, value in properties.items():
        molobj.SetProp(key, str(value))  # type: ignore[attr-defined]
    return


def unique(molobjs: List[Mol]) -> List[Mol]:
    """Return only unique molecules, based on canonical SMILES"""
    smiles_list = [molobj_to_smiles(x, canonical=True, remove_hs=True) for x in molobjs]
    _, indices = np.unique(smiles_list, return_index=True)
    return [molobjs[idx] for idx in indices]
