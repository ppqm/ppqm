import numpy as np
from rdkit import Chem

import ppqm
from ppqm import chembridge


def test_axyzc_to_molobj():

    atoms = ["C", "H", "H"]
    charge = -1
    coord = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    mol = chembridge.axyzc_to_molobj(atoms, coord, charge)

    assert mol is not None
    assert len(list(mol.GetAtoms())) == 3
    assert mol.GetNumConformers() == 1

    atoms_prime, coord_prime, charge_prime = chembridge.get_axyzc(mol, atomfmt=str)

    print(atoms)
    print(coord)
    print(charge)

    print(atoms_prime)
    print(coord_prime)
    print(charge_prime)

    assert charge_prime == charge
    assert all([a == b for a, b in zip(atoms_prime, atoms)])
    np.testing.assert_array_equal(coord_prime, coord)


def test_clean_sdf_header():
    pass


def test_conformer_set_coordinates():
    pass


def test_copy_molobj():

    smiles = "CCCCO"
    molobj = Chem.MolFromSmiles(smiles)
    molobj = ppqm.tasks.generate_conformers(molobj)
    assert molobj.GetNumConformers() > 1

    # Copy only graph from molobj
    molobj_prime = chembridge.copy_molobj(molobj)
    assert molobj_prime.GetNumConformers() == 0


def test_enumerate_stereocenters():

    smiles_prime = "F[C@@](Cl)(Br)I"  # find this
    smiles = "FC(Cl)(Br)I"  # in this
    molobj = Chem.MolFromSmiles(smiles)
    assert molobj is not None

    # Get all enuerated stereo-centers
    molobj_list = chembridge.enumerate_stereocenters(molobj)
    smiles_list = [Chem.MolToSmiles(mol) for mol in molobj_list]

    assert len(smiles_list) == 2
    assert smiles_prime in smiles_list


def test_find_max_feature():
    smiles = "CCCCCC.CCCCO"
    smiles_prime = "CCCCO"
    smiles_tau = chembridge.find_max_feature(smiles)
    assert smiles_tau == smiles_prime


def test_find_max_str():
    smiles = "CCCCC.[Cl-]"
    smiles_prime = "CCCCC"
    smiles_tau = chembridge.find_max_feature(smiles)
    assert smiles_tau == smiles_prime


def test_get_atom_charges():

    smiles = "CCC[NH+](C)C"  # n,n-dimethylpropan-1-amine
    molobj = Chem.MolFromSmiles(smiles)
    charges = chembridge.get_atom_charges(molobj)
    charges = list(charges)

    assert charges == [0, 0, 0, 1, 0, 0]


def test_get_atom_int():
    assert chembridge.get_atom_int("C") == 6
    assert chembridge.get_atom_int("c") == 6
    assert chembridge.get_atom_int("  C") == 6
    assert chembridge.get_atom_int("Cl ") == 17
    assert chembridge.get_atom_int("cl ") == 17
    assert chembridge.get_atom_int("CL") == 17


def test_get_atom_str():
    assert chembridge.get_atom_str(6) == "C"
    assert chembridge.get_atom_str(17) == "Cl"


def test_get_atoms():
    smiles = "CCC[NH+](C)C"  # n,n-dimethylpropan-1-amine
    molobj = Chem.MolFromSmiles(smiles)

    atoms = chembridge.get_atoms(molobj)
    atoms = list(atoms)
    assert atoms == [6, 6, 6, 7, 6, 6]

    atoms = chembridge.get_atoms(molobj, type=str)
    atoms = list(atoms)
    assert atoms == ["C", "C", "C", "N", "C", "C"]


def test_get_axyzc():
    smiles = "CCC[NH+](C)C"  # n,n-dimethylpropan-1-amine
    n_hydrogens = 14

    molobj = Chem.MolFromSmiles(smiles)
    molobj = ppqm.tasks.generate_conformers(molobj)

    atoms, coord, charge = chembridge.get_axyzc(molobj)

    assert list(atoms) == [6, 6, 6, 7, 6, 6] + [[1]] * n_hydrogens
    assert isinstance(coord, np.ndarray)
    assert coord.shape == (6 + n_hydrogens, 3)
    assert charge == 1


def test_get_boltzmann_weights():

    conformer_energies = [5.0, 1.0, 20.0, 0.5, 0.0]
    conformer_energies = np.asarray(conformer_energies)

    weights = chembridge.get_boltzmann_weights(conformer_energies)
    weights = np.round(weights, 2)
    assert list(weights) == [0, 0.11, 0, 0.27, 0.62]


def test_get_bonds():

    smiles = "CCC[NH+](C)C"  # n,n-dimethylpropan-1-amine
    bonds_prime = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5)]

    molobj = Chem.MolFromSmiles(smiles)
    bonds = chembridge.get_bonds(molobj)
    assert bonds == bonds_prime


def test_get_canonical_smiles():
    smiles = "C[NH+](CCC)C"  # n,n-dimethylpropan-1-amine
    smiles_prime = "CCC[NH+](C)C"  # n,n-dimethylpropan-1-amine
    assert chembridge.get_canonical_smiles(smiles) == smiles_prime


def test_get_center_of_mass():
    smiles = "C[NH+](CCC)C"  # n,n-dimethylpropan-1-amine
    molobj = Chem.MolFromSmiles(smiles)
    molobj = ppqm.tasks.generate_conformers(molobj)

    atoms, coordinates, _ = chembridge.get_axyzc(molobj)
    center = chembridge.get_center_of_mass(atoms, coordinates)
    center = np.round(center, 2)

    assert list(center) == [-0.03, 0.0, -0.01]


def test_get_dipole_moments():
    smiles = "C[NH+](CCC)C"  # n,n-dimethylpropan-1-amine
    n_conformers = 4
    molobj = Chem.MolFromSmiles(smiles)
    molobj = ppqm.tasks.generate_conformers(molobj, n_conformers=n_conformers)

    moments = chembridge.get_dipole_moments(molobj)
    assert moments is not None
    assert isinstance(moments, np.ndarray)
    assert moments.shape == (n_conformers,)


def test_get_inertia():
    pass


def test_get_inertia_diag():
    pass


def test_get_inertia_ratio():
    pass


def test_get_inertia_ratios():
    pass


def test_get_properties_from_molobj():
    pass


def test_get_sasa():
    pass


def test_get_torsions():
    pass


def test_get_undefined_stereocenters():
    pass


def test_molobj_add_conformer():
    pass


def test_molobj_check_distances():
    pass


def test_molobj_select_conformers():
    pass


def test_molobj_set_coordinates():
    pass


def test_molobjs_to_molobj():
    pass


def test_molobjs_to_properties():
    pass


def test_molobj_to_mol2():
    pass


def test_molobj_to_molobjs():
    pass


def test_molobj_to_sdfstr():
    pass


def test_molobj_to_smiles():
    pass


def test_molobj_to_svgstr():
    pass


def test_neutralize_molobj():
    pass


def test_read():
    pass


def test_read_smi():
    pass


def test_sdfstrs_to_molobjs():
    pass


def test_sdfstr_to_molobj():

    # TODO Add some properties

    sdfstr = """


  7  6  0  0  0  0  0  0  0  0999 V2000
   -0.0110    0.9628    0.0073 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.2864    1.5618   -0.0018 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.1542    2.9846    0.0071 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0021   -0.0041    0.0020 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8401    1.2428    0.8812 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.8231    1.2523   -0.8987 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.9974    3.4580    0.0016 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  4  1  0  0  0  0
  2  3  1  0  0  0  0
  2  5  1  0  0  0  0
  2  6  1  0  0  0  0
  3  7  1  0  0  0  0
M  END
$$$$


"""

    molobj = chembridge.sdfstr_to_molobj(sdfstr)

    assert molobj is not None
    assert isinstance(molobj, Chem.Mol)
    assert molobj.GetNumConformers() == 1


def test_sdfstr_to_smiles():
    pass


def test_smiles_to_molobj():
    pass


def test_set_properties_on_molobj():
    pass
