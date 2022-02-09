import pytest

from collections import ChainMap

from context import ORCA_OPTIONS, RESOURCES

from ppqm import chembridge, orca, units, tasks

TEST_ENERGIES = [
    ("O", -11.937593653418 * units.hartree_to_kcalmol),
    ("CC", -12.125402364292 * units.hartree_to_kcalmol),
    ("[NH4+]",  -7.973713141342 * units.hartree_to_kcalmol),
]

def _get_options(tmpdir):
    orca_options = {"scr": tmpdir, "n_cores": 1,"memory": 2}
    options_prime = ChainMap(orca_options, ORCA_OPTIONS)
    options_prime = dict(options_prime)
    return options_prime

@pytest.mark.parametrize("smiles, energy", TEST_ENERGIES)
def test_axyzc_optimize(smiles, energy, tmpdir):
    """ TODO: optimize not just SP """

    orca_options = _get_options(tmpdir)

    molobj = chembridge.smiles_to_molobj(smiles)
    molobj = tasks.generate_conformers(molobj, n_conformers=1)

    assert molobj is not None

    calculation_options = {"pm3": None}
    atoms, coordinates, charge = chembridge.get_axyzc(molobj, atomfmt=str)

    properties = orca.get_properties_from_axyzc(
        atoms, coordinates, charge, spin=1, options=calculation_options, **orca_options
    )

    scf_energy = properties[orca.COLUMN_SCF_ENERGY]

    assert pytest.approx(energy, 10 ** -4) == scf_energy

def test_get_mulliken_charges():

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    mulliken_charges = orca.get_mulliken_charges(lines)

    assert "mulliken_charges" in mulliken_charges

    assert len(mulliken_charges["mulliken_charges"]) == 14
    assert isinstance(mulliken_charges["mulliken_charges"][0], float)

    assert mulliken_charges["mulliken_charges"][0] == 0.220691
    assert mulliken_charges["mulliken_charges"][-1] == 0.197727

def test_get_loewdin_charges():

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    loewdin_charges = orca.get_loewdin_charges(lines)

    assert "loewdin_charges" in loewdin_charges

    assert len(loewdin_charges["loewdin_charges"]) == 14
    assert isinstance(loewdin_charges["loewdin_charges"][0], float)

    assert loewdin_charges["loewdin_charges"][0] == 0.073725
    assert loewdin_charges["loewdin_charges"][-1] == 0.095532

def test_get_hirshfeld_charges():

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    hirshfeld_charges = orca.get_hirshfeld_charges(lines)

    assert "hirshfeld_charges" in hirshfeld_charges

    assert len(hirshfeld_charges["hirshfeld_charges"]) == 14
    assert isinstance(hirshfeld_charges["hirshfeld_charges"][0], float)

    assert hirshfeld_charges["hirshfeld_charges"][0] == 0.044416
    assert hirshfeld_charges["hirshfeld_charges"][-1] == 0.183978

def test_nmr_shielding_constants():

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    shielding_constants = orca.get_nmr_shielding_constants(lines)

    assert "shielding_constants" in shielding_constants

    assert len(shielding_constants["shielding_constants"]) == 14
    assert isinstance(shielding_constants["shielding_constants"][0], float)

    assert shielding_constants["shielding_constants"][0] == 124.718
    assert shielding_constants["shielding_constants"][-1] == 30.530