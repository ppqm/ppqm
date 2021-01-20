import numpy as np

from ppqm import chembrige


def test_axyzc_to_molobj():

    atoms = ["C", "H", "H"]
    charge = -1
    coord = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    mol = chembrige.axyzc_to_molobj(atoms, coord, charge)

    assert mol is not None
    assert len(list(mol.GetAtoms())) == 3
    assert mol.GetNumConformers() == 1

    atoms_prime, coord_prime, charge_prime = chembrige.get_axyzc(mol)

    assert charge_prime == charge


def test_bonds_to_molobj():
    pass


def test_clean_sdf_header():
    pass


def test_conformer_set_coordinates():
    pass


def test_copy_molobj():
    pass


def test_enumerate_stereocenters():
    pass


def test_find_max_feature():
    pass


def test_find_max_str():
    pass


def test_get_atom_charges():
    pass


def test_get_atom_int():
    pass


def test_get_atom_str():
    pass


def test_get_atoms():
    pass


def test_get_axyzc():
    pass


def test_get_boltzmann_weights():
    pass


def test_get_bonds():
    pass


def test_get_canonical_smiles():
    pass


def test_get_center_of_mass():
    pass


def test_get_dipole_moments():
    pass


def test_get_dipole_moment():
    pass


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
    pass


def test_sdfstr_to_smiles():
    pass


def test_smiles_to_molobj():
    pass


def test_set_properties_on_molobj():
    pass
