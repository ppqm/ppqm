import logging

import pytest
from context import XTB_OPTIONS

from ppqm import chembridge, tasks, xtb

logging.basicConfig(level=logging.DEBUG)

TEST_ENERGIES = [
    ("O", -5.0705443306),
    ("CC", -7.336370430847),
    ("[NH4+]", -4.615767064363),
]

TEST_SMILES = ["O", "N"]


def _get_options(tmpdir):
    xtb_options = {"scr": tmpdir, **XTB_OPTIONS}
    return xtb_options


@pytest.mark.parametrize("smiles, energy", TEST_ENERGIES)
def test_axyzc_optimize(smiles, energy, tmpdir):

    xtb_options = _get_options(tmpdir)

    # TODO Get distances between heavy atoms
    # TODO assert distances between heavy atoms approx(distance, 0.5)
    # TODO Test distance
    # - ensure convergence
    # - ensure aang units

    molobj = chembridge.smiles_to_molobj(smiles)
    molobj = tasks.generate_conformers(molobj, n_conformers=1)

    assert molobj is not None

    atoms, coordinates, charge = chembridge.get_axyzc(molobj, atomfmt=str)

    calculation_options = {"opt": None, "gfn": 2, "gbsa": "water"}

    # TODO Get distances between heavy atoms

    properties = xtb.get_properties_from_axyzc(
        atoms, coordinates, charge, options=calculation_options, **xtb_options
    )

    total_energy = properties[xtb.COLUMN_ENERGY]
    assert xtb.COLUMN_COORD in properties
    assert pytest.approx(energy, 10 ** -2) == total_energy


@pytest.mark.parametrize("smiles, energy", TEST_ENERGIES)
def test_calc_options(smiles, energy):

    molobj = chembridge.smiles_to_molobj(smiles)
    molobj = tasks.generate_conformers(molobj, n_conformers=2)
    calculation_options = {"opt": None, "gfn": 2, "gbsa": "water"}

    # Make calculator instance
    calc = xtb.XtbCalculator(**XTB_OPTIONS)
    print(calc)

    properties_list = calc.calculate(molobj, options=calculation_options)
    print(properties_list)

    assert isinstance(properties_list, list)
    assert len(properties_list) == 2
    assert isinstance(properties_list[0], dict)
    assert xtb.COLUMN_ENERGY in properties_list[0]


# def test_parseproperties():
#
#     filename = "tests/resources/water.log"
#
#     with open(filename, "r") as f:
#         lines = f.readlines()
#
#     properties = xtb.read_properties(lines)
#     total_energy = properties[xtb.COLUMN_ENERGY]
#     total_dipole = properties[xtb.COLUMN_DIPOLE]
#
#     # Check energy
#     assert pytest.approx(-5.082177326958, 10 ** -7) == total_energy
#
#     # Check dipole moment
#     assert pytest.approx(2.422, 10 ** -4) == total_dipole
#
#     return


# def test_single_point_charges():
#     # Create molecule with a minus charge
#     # Calculate the energy
#
#     smi = "C[O-]"
#
#     molobj = Chem.MolFromSmiles(smi)
#     molobj = molpipe.cheminfo.generate_conformers(molobj, n_conformers=2)
#     properties_list = xtb.get_properties_from_molobj(molobj)
#     assert xtb.COLUMN_ENERGY in properties_list[0]
#     assert isinstance(properties_list[0][xtb.COLUMN_ENERGY], float)
#     assert len(properties_list) == 2
#
#
# def test_failed_scf():
#
#     assert True
#
#     return
#
#
# def test_failed_optimize():
#
#     assert True
#
#     return
#
#
# def test_parallel():
#
#     assert True
#
#     return
#


def test_parse_sum_table():

    sumtable = """
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         ::                     SUMMARY                     ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         :: total energy              -4.172643383317 Eh    ::
         :: total w/o Gsasa/hb        -4.172990490529 Eh    ::
         :: gradient norm              0.035613904556 Eh/a0 ::
         :: HOMO-LUMO gap             16.021567607307 eV    ::
         ::.................................................::
         :: SCC energy                -4.232712843426 Eh    ::
         :: -> isotropic ES            0.002083179949 Eh    ::
         :: -> anisotropic ES          0.002729145429 Eh    ::
         :: -> anisotropic XC          0.004205876144 Eh    ::
         :: -> dispersion             -0.000660899375 Eh    ::
         :: -> Gsolv                   0.000347007313 Eh    ::
         ::    -> Gborn               -0.000000099899 Eh    ::
         ::    -> Gsasa               -0.001315296325 Eh    ::
         ::    -> Ghb                 -0.000195039589 Eh    ::
         ::    -> Gshift               0.001857443127 Eh    ::
         :: repulsion energy           0.060069453613 Eh    ::
         :: add. restraining           0.000000000000 Eh    ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
"""

    lines = sumtable.split("\n")
    properties = xtb.parse_sum_table(lines)
    assert type(properties) == dict
    assert properties["gsolv"] == 0.000347007313


# def test_multiple_solvents():
#     """
#     The ALPB solvation model is implemented in version 6.3.3 or newer, use
#     --gbsa in older versions instead
#
#     https://xtb-docs.readthedocs.io/en/latest/gbsa.html
#
#     """
#
#     kwargs = {"show_progress": False}
#
#     first_gen_options = {"gbsa": "water", "opt": None}
#
#     second_gen_options = {"alpb": "water", "opt": None}
#
#     smi = "C"
#     molobj = Chem.MolFromSmiles(smi)
#     molobj = molpipe.cheminfo.generate_conformers(molobj, n_conformers=1)
#
#     assert molobj.GetNumConformers() == 1
#
#     properties = molpipe.xtb.get_properties_from_molobj(
#         molobj, options=first_gen_options, **kwargs
#     )
#
#     properties = properties[0]
#
#     assert "gsolv" in properties
#     assert pytest.approx(0.00034, abs=1e-4) == properties["gsolv"]
#
#     properties = molpipe.xtb.get_properties_from_molobj(
#         molobj, options=second_gen_options, **kwargs
#     )
#     properties = properties[0]
#
#     print(properties)
#
#     assert "gsolv" in properties
#     assert pytest.approx(0.000243700318, abs=1e-4) == properties["gsolv"]
#
#
# def test_read_orbitals():
#
#     # 61        2.0000           -0.3707448             -10.0885
#     # 62        2.0000           -0.3635471              -9.8926
#     # 63        2.0000           -0.3540913              -9.6353 (HOMO)
#     # 64                         -0.2808508              -7.6423 (LUMO)
#     # 65                         -0.2674644              -7.2781
#     # 66                         -0.2487986              -6.7702
#
#     # 4        2.0000           -0.4396701             -11.9640 (HOMO)
#     # 5                          0.1005728               2.7367 (LUMO)
#     # 6                          0.2725561               7.4166
#
#     # big molecule
#     logfilename = RESOURCES / "chembl3586573.log"
#     with open(logfilename, "r") as f:
#         lines = f.readlines()
#     properties = molpipe.xtb.read_properties_orbitals(lines)
#
#     assert properties["homo"] == -0.3540913
#     assert properties["lumo"] == -0.2808508
#     assert properties["lumo+1"] == -0.2674644
#     assert properties["homo-1"] == -0.3635471
#
#     # small molecule
#     logfilename = RESOURCES / "water.log"
#     with open(logfilename, "r") as f:
#         lines = f.readlines()
#     properties = molpipe.xtb.read_properties_orbitals(lines)
#
#     print(properties)
#
#     assert properties["homo"] == -0.4396701
#     assert properties["lumo"] == 0.1005728
#     assert properties["lumo+1"] == 0.2725561
#
#
# def test_read_fukui():
#
#     # #        f(+)     f(-)     f(0)
#     # 1O      -0.086   -0.598   -0.342
#     # 2H      -0.457   -0.201   -0.329
#     # 3H      -0.457   -0.201   -0.329
#     logfilename = RESOURCES / "water_fukui.log"
#
#     with open(logfilename, "r") as f:
#         lines = f.readlines()
#
#     properties = molpipe.xtb.read_properties_fukui(lines)
#
#     assert "f_plus" in properties
#     assert "f_minus" in properties
#     assert "f_zero" in properties
#     assert len(properties.get("f_plus")) == 3
#     assert properties.get("f_plus")[0] == -0.086
#
#
# def test_read_omega():
#
#     # Global electrophilicity index (eV):    0.0058
#
#     logfilename = RESOURCES / "water_omega.log"
#
#     with open(logfilename, "r") as f:
#         lines = f.readlines()
#
#     properties = molpipe.xtb.read_properties_omega(lines)
#     assert isinstance(properties, dict)
#     assert "global_electrophilicity_index" in properties
#     assert properties.get("global_electrophilicity_index") == 0.0058
#
#     properties = molpipe.xtb.read_properties(lines, options={"vomega": None})
#     assert isinstance(properties, dict)
#     assert "global_electrophilicity_index" in properties
#     assert properties.get("global_electrophilicity_index") == 0.0058
#
#
# def test_calculate_fukui():
#
#     # Chembl
#     chembl3586573 = "CCN1C(=O)CNc2ncc(-c3ccc(-c4nc[nH]n4)nc3C)nc21"
#     molobj = Chem.MolFromSmiles(chembl3586573)
#
#     # Embed 3D
#     n_conformers = 1
#     molobj = molpipe.cheminfo.generate_conformers(molobj, n_conformers=n_conformers)
#     assert molobj.GetNumConformers() == n_conformers
#     n_atoms = molobj.GetNumAtoms()
#
#     # xtb options
#     kwargs = {"show_progress": False, "n_cores": 1}
#
#     # Calc options
#     solvent = "h2o"
#     optimize_options = {
#         "gfn": 2,
#         "alpb": solvent,
#         "opt": None,
#     }
#
#     fukui_options = {
#         "gfn": 2,
#         "alpb": solvent,
#         "vfukui": None,
#     }
#
#     # Optimize structure
#     properties_list = molpipe.xtb.get_properties_from_molobj(
#         molobj, options=optimize_options, **kwargs
#     )
#     properties = properties_list[0]
#     assert molpipe.xtb.COLUMN_COORD in properties
#
#     # Set structure
#     coord = properties.get(molpipe.xtb.COLUMN_COORD)
#     molpipe.cheminfo.set_coordinates(molobj, coord)
#
#     # Get fukui
#     properties_list = molpipe.xtb.get_properties_from_molobj(
#         molobj, options=fukui_options, **kwargs
#     )
#     assert isinstance(properties_list, list)
#     assert len(properties_list) == n_conformers
#
#     properties = properties_list[0]
#
#     assert "f_plus" in properties
#     assert len(properties.get("f_plus")) == n_atoms
#
#
# def test_calculate_electrophilicity():
#
#     # Chembl
#     chembl3586573 = "CCN1C(=O)CNc2ncc(-c3ccc(-c4nc[nH]n4)nc3C)nc21"
#     molobj = Chem.MolFromSmiles(chembl3586573)
#
#     # Embed 3D
#     n_conformers = 1
#     molobj = molpipe.cheminfo.generate_conformers(molobj, n_conformers=n_conformers)
#     assert molobj.GetNumConformers() == n_conformers
#     molobj.GetNumAtoms()
#
#     # xtb options
#     kwargs = {"show_progress": False, "n_cores": 1}
#
#     # Calc options
#     solvent = "h2o"
#     optimize_options = {
#         "gfn": 2,
#         "alpb": solvent,
#         "opt": None,
#     }
#
#     omega_options = {
#         "gfn": 2,
#         "alpb": solvent,
#         "vomega": None,
#     }
#
#     # Optimize structure
#     properties_list = molpipe.xtb.get_properties_from_molobj(
#         molobj, options=optimize_options, **kwargs
#     )
#     properties = properties_list[0]
#     assert molpipe.xtb.COLUMN_COORD in properties
#
#     # Set structure
#     coord = properties.get(molpipe.xtb.COLUMN_COORD)
#     molpipe.cheminfo.set_coordinates(molobj, coord)
#
#     # Get electrophilicity (vomega)
#     properties_list = molpipe.xtb.get_properties_from_molobj(
#         molobj, options=omega_options, **kwargs
#     )
#     assert isinstance(properties_list, list)
#     assert len(properties_list) == n_conformers
#
#     properties = properties_list[0]
#
#     print(properties)
#
#     assert "global_electrophilicity_index" in properties
#     assert isinstance(properties.get("global_electrophilicity_index"), float)
#     assert properties.get("global_electrophilicity_index") == pytest.approx(2, rel=1)
#
#
# def test_get_all_properties():
#
#     filename = RESOURCES / "chembl3586573_with_conformers.sdf"
#     molobjs = molpipe.cheminfo.read(filename)
#     molobjs = list(molobjs)
#     molobj = molpipe.cheminfo.molobjs_to_molobj(molobjs)
#
#     assert molobj is not None
#     assert molobj.GetNumConformers() == 13
#
#     kwargs = {"n_cores": 1, "show_progress": False}
#
#     properties = molpipe.pipe.get_all_properties_xtb(molobj, solvent="h2o", xtb_options=kwargs)
#
#     print(properties)
#
#     assert properties is not None
#     assert isinstance(properties, list)
#     assert len(properties) == molobj.GetNumConformers()
#     assert isinstance(properties[0], dict)
#
#     assert "h2o_f_plus" in properties[0]
#     assert "h2o_global_electrophilicity_index" in properties[0]
#
#     return
#
#
# if __name__ == "__main__":
#     test_get_all_properties()
