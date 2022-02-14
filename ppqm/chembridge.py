import copy
import gzip
from io import StringIO
from typing import Any, Dict, List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, Mol, rdFreeSASA, rdmolops
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize

from ppqm import units

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


def axyzc_to_molobj(atoms: List[str], coord: np.array, charge: int) -> Mol:
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
    mol = write_mol.GetMol()

    # Set coordinates / Conformer
    conformer = Chem.Conformer(n_atoms)
    conformer_set_coordinates(conformer, coord)
    mol.AddConformer(conformer, assignId=True)

    # Set charge on a random atom
    atoms = list(mol.GetAtoms())
    for atom in atoms:
        if atom.GetAtomicNum() != 1:
            break

    atom.SetFormalCharge(charge)

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


def conformer_set_coordinates(conformer: Chem.Conformer, coordinates: np.ndarray) -> None:
    for i, pos in enumerate(coordinates):
        conformer.SetAtomPosition(i, pos)


def copy_molobj(molobj: Mol) -> Mol:
    """ Copy molobj graph, without conformers """
    # The boolean signifies a fast copy, e.g. no conformers
    molobj = Chem.Mol(molobj, True)
    return molobj


def enumerate_stereocenters(
    molobj: Mol,
    nbody: int = 2,
) -> List[Mol]:
    """ Find all un-assigned stereocenteres and assign them """

    properties = get_properties_from_molobj(molobj)

    max_unassigned_stereo_centers = 3

    stereo_options = StereoEnumerationOptions(
        tryEmbedding=True,
        onlyUnassigned=True,
        maxIsomers=max_unassigned_stereo_centers,
        rand=1,
    )

    # NOTE try/except to capture
    # Pre-condition Violation\n\tStereo atoms should be specified before
    # specifying CIS/TRANS bond stereochemistry

    try:
        enumerator = EnumerateStereoisomers(molobj, options=stereo_options)
        isomers = list(enumerator)
    except RuntimeError:
        return [molobj]

    # The isomer can contain steorecenter enumerating that is non-unique
    isomers = unique(isomers)

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


def get_atom_charges(molobj: Mol) -> List[int]:
    """ Get atom charges from molobj """
    atoms = molobj.GetAtoms()
    charges = [atom.GetFormalCharge() for atom in atoms]
    charges = np.array(charges)
    return charges


def get_atom_int(atmstr: str) -> int:
    """ Get atom number from atom label """
    atom = atmstr.strip().lower()
    atom = ATOM_LIST.index(atom) + 1
    return atom


def get_atom_str(iatm: int) -> str:
    """ Get atom label from atom number """

    atom = ATOM_LIST[iatm - 1]
    atom = atom.capitalize()

    return atom


def get_atoms(mol: Mol, type=int) -> List[Any]:
    """ Get atoms from molecule in either int or str format """

    atoms = mol.GetAtoms()

    if type == int:
        atoms = [a.GetAtomicNum() for a in atoms]

    elif type == str:
        atoms = [a.GetSymbol() for a in atoms]

    else:
        return atoms

    atoms = np.array(atoms)

    return atoms


def get_axyzc(molobj: Mol, confid: int = -1, atomfmt=int) -> Tuple[List[Any], np.ndarray, int]:
    """ Get atoms, XYZ coordinates and formal charge of a molecule """
    conformer = molobj.GetConformer(id=confid)
    coordinates = conformer.GetPositions()
    coordinates = np.array(coordinates)
    atoms = get_atoms(molobj, type=atomfmt)
    charge = get_charge(molobj)
    return atoms, coordinates, charge


def get_charge(molobj: Mol) -> int:
    charge = rdmolops.GetFormalCharge(molobj)
    return charge


def get_coordinates(molobj: Mol, confid=-1) -> np.ndarray:
    """ """
    confid = int(confid)  # rdkit needs int type
    conformer = molobj.GetConformer(id=confid)
    coordinates = conformer.GetPositions()
    coordinates = np.asarray(coordinates)
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
    """ Get all bonds from molobj """

    bonds = molobj.GetBonds()

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
    """ Translate smiles into a canonical form """
    molobj = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(molobj, canonical=True)
    return smiles


def get_center_of_mass(atoms: List[int], coordinates: np.ndarray) -> np.ndarray:
    """ Calculate center of mass """

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


def get_dipole_moments(molobj):
    """
    Compute dipole moment for all conformers, using Gasteiger charges and
    coordinates from conformers

    Expects molobj to contain conformers

    """

    # Conformer independent
    AllChem.ComputeGasteigerCharges(molobj)

    atoms = molobj.GetAtoms()
    atoms_int = [atom.GetAtomicNum() for atom in atoms]
    atoms_charge = [atom.GetDoubleProp("_GasteigerCharge") for atom in atoms]
    atoms_charge = np.array(atoms_charge)

    # Calculate moments for each conformer
    moments = []
    for conformer in molobj.GetConformers():
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)

        total_moment = get_dipole_moment(atoms_int, coordinates, atoms_charge)
        moments.append(total_moment)

    moments = np.array(moments)

    return moments


def get_dipole_moment(atoms, coordinates, charges, is_centered=False):
    """

    from wikipedia:
    For a charged molecule the center of charge should be the reference point
    instead of the center of mass.

    """

    total_charge = np.sum(charges)
    total_charge = int(total_charge)

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
    total_moment = np.linalg.norm(xyz)

    return total_moment


def get_inertia(atoms, coordinates):
    """ Calculate inertia moments """

    com = get_center_of_mass(atoms, coordinates)

    coordinates -= com

    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    Z = coordinates[:, 2]

    rxx = Y ** 2 + Z ** 2
    ryy = X ** 2 + Z ** 2
    rzz = X ** 2 + Y ** 2

    Ixx = atoms * rxx
    Iyy = atoms * ryy
    Izz = atoms * rzz

    Ixy = atoms * Y * X
    Ixz = atoms * X * Z
    Iyz = atoms * Y * Z

    Ixx = np.sum(Ixx)
    Iyy = np.sum(Iyy)
    Izz = np.sum(Izz)

    Ixy = np.sum(Ixy)
    Ixz = np.sum(Ixz)
    Iyz = np.sum(Iyz)

    inertia = np.zeros((3, 3))

    inertia[0, 0] = Ixx
    inertia[1, 1] = Iyy
    inertia[2, 2] = Izz

    inertia[0, 1] = -Ixy
    inertia[1, 0] = -Ixy
    inertia[0, 2] = -Ixz
    inertia[2, 0] = -Ixz
    inertia[1, 2] = -Iyz
    inertia[2, 1] = -Iyz

    w, _ = np.linalg.eig(inertia)

    return w


def get_inertia_diag(atoms, coordinates):
    """ Calcualte the inertia diagonal vector """

    com = get_center_of_mass(atoms, coordinates)

    coordinates -= com

    X = coordinates[:, 0]
    Y = coordinates[:, 1]
    Z = coordinates[:, 2]

    rx2 = Y ** 2 + Z ** 2
    ry2 = X ** 2 + Z ** 2
    rz2 = X ** 2 + Y ** 2

    Ix = atoms * rx2
    Iy = atoms * ry2
    Iz = atoms * rz2

    Ix = np.sum(Ix)
    Iy = np.sum(Iy)
    Iz = np.sum(Iz)

    inertia = np.zeros(3)
    inertia[0] = Ix
    inertia[1] = Iy
    inertia[2] = Iz

    return inertia


def get_inertia_ratio(inertia):
    """ Sort intertia digonal and calculate the shape ratio """

    inertia.sort()

    ratio = np.zeros(2)
    ratio[0] = inertia[0] / inertia[2]
    ratio[1] = inertia[1] / inertia[2]

    return ratio


def get_inertia_ratios(molobj):
    """ Get inertia ratios for all conformers """

    atoms = molobj.GetAtoms()
    atoms_int = [atom.GetAtomicNum() for atom in atoms]
    atoms_int = np.array(atoms_int)

    # Calculate moments for each conformer
    ratios = []
    for conformer in molobj.GetConformers():
        coordinates = conformer.GetPositions()
        coordinates = np.array(coordinates)

        inertia = get_inertia_diag(atoms_int, coordinates)
        ratio = get_inertia_ratio(inertia)
        ratios.append(ratio)

    ratios = np.array(ratios)

    return ratios


def get_properties_from_molobj(molobj):
    """ Get properties from molobj """
    properties = molobj.GetPropsAsDict()
    return properties


def get_sasa(molobj, extra_radius=0.0):
    """Get solvent accessible surface area per atom

    :param molobj: Molecule with 3D conformers
    :param extra_radius: Constant addition to the atom radii's

    :return sasa: List of area, for each conformer

    """

    radii = [PTABLE.GetRvdw(atom.GetAtomicNum()) for atom in molobj.GetAtoms()]

    n = molobj.GetNumConformers()

    radii = [r + extra_radius for r in radii]
    sasas = np.zeros(n)

    for i in range(n):
        sasa = rdFreeSASA.CalcSASA(molobj, radii, confIdx=i)
        sasas[i] = sasa

    return sasas


def get_torsions(mol):
    """
    Get indices of all torsion pairs All heavy atoms.
    One end can be Hydrogen.
    """

    any_atom = "[*]"
    smarts = [any_atom, any_atom, any_atom, any_atom]
    smarts = "~".join(smarts)

    atoms = get_atoms(mol, type=str)

    idxs = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
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
    """ Count number of undefined steorecenter in molobj """
    chiral_centers = dict(Chem.FindMolChiralCenters(molobj, includeUnassigned=True))
    n_undefined_centers = sum(1 for (x, y) in chiral_centers.items() if y == "?")
    return n_undefined_centers


def molobj_add_conformer(molobj, coordinates):
    """ Append coordinates as a new conformer to molobj """
    conf = Chem.Conformer(len(coordinates))
    for i, coordinate in enumerate(coordinates):
        conf.SetAtomPosition(i, coordinate)
    molobj.AddConformer(conf, assignId=True)


def molobj_check_distances(molobj, min_cutoff=0.001, max_cutoff=3.0):
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

    status = np.array(status)

    return status


def molobj_select_conformers(molobj, idxs):
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

        conf = molobj.GetConformer(id=idx)
        molobj_prime.AddConformer(conf, assignId=True)

    return molobj_prime


def molobj_set_coordinates(molobj, coordinates, confid=-1):
    conformer = molobj.GetConformer(id=confid)
    conformer_set_coordinates(conformer, coordinates)


def molobjs_to_molobj(molobjs):
    """
    take list of molobjs and merge into molobj with conformers

    IMPORTANT: expects all molobjs to be same graph and same atom order!
    """

    molobj = copy_molobj(molobjs[0])
    n_molecules = len(molobjs)

    atoms = get_atoms(molobjs[0], type=int)
    atoms = list(atoms)

    for idx in range(n_molecules):

        # Test we don't mix and match molecules
        assert molobjs[idx].GetNumConformers() == 1
        atoms_prime = get_atoms(molobjs[idx])
        atoms_prime = list(atoms_prime)
        assert atoms == atoms_prime, "Cannot merge two different molecules"

        conf = molobjs[idx].GetConformer(id=-1)
        molobj.AddConformer(conf, assignId=True)

    return molobj


def molobjs_to_properties(molobjs: List[Mol]) -> Dict[str, List[Any]]:
    """Return a dictionary of every property found in the molobj.

    :param molobjs: Iter[Mol] List of molobjs
    :return properties: Dict[Str, List[Value]]
    """

    all_properties = []
    keys = []

    for molobj in molobjs:
        properties = molobj.GetPropsAsDict()
        all_properties.append(properties)

        keys += list(properties.keys())

    keys = np.unique(keys)

    rtn_values = {key: [] for key in keys}

    for properties in all_properties:
        for key in keys:

            if key in properties:
                value = properties[key]
            else:
                value = None

            rtn_values[key].append(value)

    return rtn_values


def molobj_to_mol2(molobj, charges=None):
    """
    https://www.mdanalysis.org/docs/_modules/MDAnalysis/coordinates/MOL2.html
    """

    # Bonds
    bond_lines = ["@<TRIPOS>BOND"]
    bond_fmt = "{0:>5} {1:>5} {2:>5} {3:>2}"
    bonds = list(molobj.GetBonds())
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
    bond_lines = "\n".join(bond_lines)

    # Atoms
    atom_lines = ["@<TRIPOS>ATOM"]
    atom_fmt = "{0:>4} {1:>4} {2:>13.4f} {3:>9.4f} {4:>9.4f} {5:>4} {6} {7} {8:>7.4f}"
    atoms = list(molobj.GetAtoms())
    atoms_int = [atom.GetAtomicNum() for atom in atoms]
    atoms_str = [atom.GetSymbol() for atom in atoms]
    atoms_int = np.array(atoms_int)
    n_atoms = len(atoms)
    conformer = molobj.GetConformer()
    coordinates = conformer.GetPositions()
    coordinates = np.array(coordinates)
    np.unique(atoms_int)

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
    atom_lines = "\n".join(atom_lines)

    # Complete
    checksumstr = f"{n_atoms} {n_bonds} 0 0 0"
    head_lines = ["@<TRIPOS>MOLECULE", "TITLE"]
    head_lines += [checksumstr, "SMALL", "MULLIKEN_CHARGES", "NAME"]
    head_lines.append("")
    head_lines = "\n".join(head_lines)

    rtnstr = head_lines + atom_lines + bond_lines

    return rtnstr


def molobj_to_molobjs(molobj):
    """ Expand a molobj conformer into a list of molobjs """

    molobj_prime = copy_molobj(molobj)

    molobjs = []

    for idx, conf in enumerate(molobj.GetConformers()):

        molobj_psi = copy.deepcopy(molobj_prime)
        molobj_psi.AddConformer(conf, assignId=True)
        molobjs.append(molobj_psi)

    return molobjs


def molobj_to_sdfstr(mol, return_list=False, use_v3000=False, include_properties=False):
    """ Get SDF string from Mol """

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

    if return_list:
        return txts

    txts = "$$$$\n".join(txts)

    return txts


def molobj_to_smiles(
    molobj,
    remove_hs=True,
    sanitize=True,
    canonical=True,
    kekulize=False,
    remove_stereo=False,
):

    if molobj is None:
        return None

    if remove_stereo:
        rdmolops.RemoveStereochemistry(molobj)

    if sanitize:
        Chem.SanitizeMol(molobj)

    if kekulize:
        Chem.Kekulize(molobj, clearAromaticFlags=True)

    if remove_hs:
        molobj = Chem.RemoveHs(molobj)

    smiles = Chem.MolToSmiles(molobj, canonical=canonical, kekuleSmiles=kekulize)

    return smiles


def molobj_to_svgstr(molobj, use_2d=True, highlights=None, pretty=False, removeHs=False):
    """
    Returns SVG in string format
    """

    if removeHs:
        molobj = Chem.RemoveHs(molobj)

    if use_2d:
        molobj = copy_molobj(molobj)
        AllChem.Compute2DCoords(molobj)

    svg = Draw.MolsToGridImage(
        [molobj],
        molsPerRow=1,
        subImgSize=(400, 400),
        useSVG=True,
        highlightAtomLists=[highlights],
    )

    svg = svg.replace("xmlns:svg", "xmlns")

    if pretty:

        svg = svg.split("\n")

        for i, line in enumerate(svg):

            # Atom letters
            if "text" in line:

                replacetext = "font-size"
                borderline = "fill:none;fill-opacity:1;stroke:#FFFFFF;stroke-width:10px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1;"

                # Add border to text
                border_text = line
                border_text = border_text.replace("stroke:none;", "")
                border_text = border_text.replace(replacetext, borderline + replacetext)

                svg[i] = border_text + "\n" + line

                continue

            if "path" in line:

                # thicker lines
                line = line.replace("stroke-width:2px", "stroke-width:3px")
                svg[i] = line

        svg = "\n".join(svg)

    svg = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")

    return svg


def neutralize_molobj(molobj):
    """
    Get a neutral protonation state of molobj and remove explicit hydrogens
    """

    total_charge = rdmolops.GetFormalCharge(molobj)

    if total_charge == 0:
        return molobj

    molobj = rdMolStandardize.ChargeParent(molobj)
    return molobj


def read(filename, remove_hs=False, sanitize=True):
    """
    General function to read files with different extensions and return molobjs

    .sdf
    .sdf.gz
    .smi
    .smi.gz

    :return molobjs:
    """

    filename = str(filename)

    if filename.endswith(".gz"):
        ext = filename.split(".")[-2:]
        ext = ".".join(ext)
    else:
        ext = filename.split(".")[-1]

    if ext == "sdf" or ext == "mol":

        suppl = Chem.SDMolSupplier(filename, removeHs=remove_hs, sanitize=sanitize)

    elif ext == "sdf.gz":

        fobj = gzip.open(filename)
        suppl = Chem.ForwardSDMolSupplier(fobj, removeHs=remove_hs, sanitize=sanitize)

    elif ext == "smi":

        f = open(filename, "r")
        suppl = read_smi(f)

    elif ext == "smi.gz":

        fobj = gzip.open(filename)
        suppl = read_smi(f)

    else:
        print("could not read file:", filename)
        quit()

    return suppl


def read_smi(f, includes_name=False):
    """
    Read smiles and yield generated molobjs with 2D coords

    :param f: iterable, either list or file
    :yields molobjs: Iteration of molobjs
    """

    if isinstance(f, str):
        f = f.split("\n")

    for line in f:
        line = line.strip()

        if includes_name:
            line = line.split()
            assert len(line) == 2
            name = line[1]
            line = line[0]

        molobj = smiles_to_molobj(line)

        # Compute 2D coords by default, to keep stereo
        AllChem.Compute2DCoords(molobj)

        if includes_name:
            yield molobj, name

        else:
            yield molobj

    return


def sdfstrs_to_molobjs(sdfs: str, remove_hs=False):
    """

    From a string of multiple SDF structures
    return a List of Mol objs
    """

    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdfs, removeHs=remove_hs)
    molobjs = [mol for mol in suppl]

    return molobjs


def sdfstr_to_molobj(sdfstr, remove_hs=False, embed_properties=True):
    """Convert SDF string to Mol

    TODO Fix suppl to
    suppl = chem sdfsupplier
    suppl.SetData(str, **kwargs)
    molobj = next(suppl)

    """

    suppl = Chem.SDMolSupplier()
    suppl.SetData(sdfstr, removeHs=remove_hs)
    try:
        molobj = next(suppl)
    except StopIteration:
        molobj = None

    # molobj = Chem.MolFromMolBlock(sdfstr, removeHs=remove_hs)
    #
    # if embed_properties:
    #     properties = get_properties_from_sdf(sdfstr)
    #
    #     for key in properties.keys():
    #         molobj.SetProp(key, properties[key])

    return molobj


def sdfstr_to_smiles(sdfstr, remove_hs=False):
    """ SDF to SMILES converter """
    mol = Chem.MolFromMolBlock(sdfstr, removeHs=remove_hs)
    smiles = Chem.MolToSmiles(mol)
    return smiles


def smiles_to_molobj(smiles, compute_2d=False, add_hydrogens=True):

    molobj = Chem.MolFromSmiles(smiles)

    if molobj is None:
        return None

    if add_hydrogens:
        molobj = rdmolops.AddHs(molobj, addCoords=True)

    if compute_2d:
        AllChem.Compute2DCoords(molobj)

    return molobj


def set_properties_on_molobj(molobj, properties):
    """ incomplete """
    for key, value in properties.items():
        molobj.SetProp(key, str(value))
    return


def unique(molobjs: List[Mol]) -> List[Mol]:
    """ Return only unique molecules, based on canonical SMILES """
    smiles_list = [molobj_to_smiles(x, canonical=True, remove_hs=True) for x in molobjs]
    _, indices = np.unique(smiles_list, return_index=True)
    return [molobjs[idx] for idx in indices]
