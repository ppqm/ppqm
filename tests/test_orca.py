from collections import ChainMap

import pytest
from context import ORCA_OPTIONS, RESOURCES
from rdkit import Chem

from ppqm import chembridge, orca, tasks, units

TEST_ENERGIES_PM3 = [
    ("O", -11.935809225486 * units.hartree_to_kcalmol),
    ("CC", -12.124869353328 * units.hartree_to_kcalmol),
    ("[NH4+]", -7.972867788142 * units.hartree_to_kcalmol),
]

TEST_ENERGIES_B3LYP = [
    ("O", -76.328663632482 * units.hartree_to_kcalmol),
    ("CC", -79.705551996245 * units.hartree_to_kcalmol),
    ("[NH4+]", -56.950888890358 * units.hartree_to_kcalmol),
]


def _get_options(tmp_path):
    orca_options = {"scr": tmp_path, "n_cores": 1, "memory": 2, "keep_files": True}
    options_prime = ChainMap(orca_options, ORCA_OPTIONS)
    options_prime = dict(options_prime)
    return options_prime


def test_parallel():
    smiles = "C(C(=O)O)N"  # I like glycine
    molobj = Chem.MolFromSmiles(smiles)

    orca_options = {
        "scr": "./_tmp_directory_",  # Where should the calculations happen?
        "cmd": "orca",  # Where is the binary executable/command?
        "n_cores": 8,  # How many cores to use?
        "show_progress": True,  # Show progressbar during calculation
    }

    calc = orca.OrcaCalculator(**orca_options)

    # Calculate values for molecule in water
    calculation_option = {
        "B3LYP": None,
        "def2-SVP": None,
        "D3BJ": None,
        "CPCM": "water",
        "RIJCOSX": None,
        "def2/J": None,
        "Grid4": None,
        "GridX4": None,
    }

    # generate conformers
    molobj_conf = tasks.generate_conformers(molobj, max_conformers=2)

    # calculate energy of conformers
    results = calc.calculate(molobj_conf, calculation_option)

    # test for some values
    scf_energy = results[1]["scf_energy"]
    mulliken_charge = results[1]["mulliken_charges"][0]

    assert pytest.approx(scf_energy, 10 ** -7) == -178251.589166
    assert mulliken_charge == 0.111094


@pytest.mark.parametrize("smiles, energy", TEST_ENERGIES_PM3)
def test_axyzc_optimize_pm3(smiles, energy, tmp_path):
    """ TODO: optimize not just SP """

    orca_options = _get_options(tmp_path)

    molobj = chembridge.smiles_to_molobj(smiles)
    molobj = tasks.generate_conformers(molobj, n_conformers=1)

    assert molobj is not None

    calculation_options = {"pm3": None}

    atoms, coordinates, charge = chembridge.get_axyzc(molobj, atomfmt=str)

    properties = orca.get_properties_from_axyzc(
        atoms, coordinates, charge, spin=1, options=calculation_options, **orca_options
    )

    assert properties is not None

    scf_energy = properties[orca.COLUMN_SCF_ENERGY]

    assert pytest.approx(energy, 10 ** -7) == scf_energy


@pytest.mark.parametrize("smiles, energy", TEST_ENERGIES_B3LYP)
def test_axyzc_optimize_b3lyp(smiles, energy, tmp_path):
    """ TODO: optimize not just SP """

    orca_options = _get_options(tmp_path)

    molobj = chembridge.smiles_to_molobj(smiles)
    molobj = tasks.generate_conformers(molobj, n_conformers=1)

    assert molobj is not None

    calculation_options = {
        "B3LYP": None,
        "def2-SVP": None,
        "D3BJ": None,
        "CPCM": "water",
        "RIJCOSX": None,
        "def2/J": None,
        "Grid4": None,
        "GridX4": None,
    }
    atoms, coordinates, charge = chembridge.get_axyzc(molobj, atomfmt=str)

    properties = orca.get_properties_from_axyzc(
        atoms, coordinates, charge, spin=1, options=calculation_options, **orca_options
    )

    assert properties is not None

    scf_energy = properties[orca.COLUMN_SCF_ENERGY]

    assert pytest.approx(energy, 10 ** -7) == scf_energy


def test_get_mulliken_charges():

    serine_num_atoms = 14

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    mulliken_charges = orca.get_mulliken_charges(lines, serine_num_atoms)

    assert mulliken_charges is not None

    assert "mulliken_charges" in mulliken_charges

    assert len(mulliken_charges["mulliken_charges"]) == serine_num_atoms
    assert isinstance(mulliken_charges["mulliken_charges"][0], float)

    assert mulliken_charges["mulliken_charges"][0] == 0.220691
    assert mulliken_charges["mulliken_charges"][-1] == 0.197727


def test_get_loewdin_charges():

    serine_num_atoms = 14

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    loewdin_charges = orca.get_loewdin_charges(lines, serine_num_atoms)

    assert loewdin_charges is not None

    assert "loewdin_charges" in loewdin_charges

    assert len(loewdin_charges["loewdin_charges"]) == serine_num_atoms
    assert isinstance(loewdin_charges["loewdin_charges"][0], float)

    assert loewdin_charges["loewdin_charges"][0] == 0.073725
    assert loewdin_charges["loewdin_charges"][-1] == 0.095532


def test_get_hirshfeld_charges():

    serine_num_atoms = 14

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    hirshfeld_charges = orca.get_hirshfeld_charges(lines, serine_num_atoms)

    assert hirshfeld_charges is not None

    assert "hirshfeld_charges" in hirshfeld_charges

    assert len(hirshfeld_charges["hirshfeld_charges"]) == serine_num_atoms
    assert isinstance(hirshfeld_charges["hirshfeld_charges"][0], float)

    assert hirshfeld_charges["hirshfeld_charges"][0] == 0.044416
    assert hirshfeld_charges["hirshfeld_charges"][-1] == 0.183978


def test_nmr_shielding_constants():

    serine_num_atoms = 14

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    shielding_constants = orca.get_nmr_shielding_constants(lines, serine_num_atoms)

    assert shielding_constants is not None

    assert "shielding_constants" in shielding_constants

    assert len(shielding_constants["shielding_constants"]) == serine_num_atoms
    assert isinstance(shielding_constants["shielding_constants"][0], float)

    assert shielding_constants["shielding_constants"][0] == 124.718
    assert shielding_constants["shielding_constants"][-1] == 30.530


def test_read_properties():

    serine_num_atoms = 14
    options = {
        "B3LYP": None,
        "def2-SVP": None,
        "D3BJ": None,
        "Hirshfeld": None,
        "CPCM": "water",
        "RIJCOSX": None,
        "def2/J": None,
        "Grid4": None,
        "GridX4": None,
        "NMR": None,
        "def2/JK": None,
    }

    logfilename = RESOURCES / "orca/serine.out"
    with open(logfilename) as f:
        lines = f.readlines()

    properties_dict = orca.read_properties(lines, serine_num_atoms, options)

    assert len(properties_dict) == 5

    # check if the read properties match expected values
    assert properties_dict["shielding_constants"][0] == 124.718
    assert properties_dict["hirshfeld_charges"][0] == 0.044416
    assert properties_dict["mulliken_charges"][0] == 0.220691


def test_read_properties_compromised_file():

    serine_num_atoms = 14
    options = {
        "B3LYP": None,
        "def2-SVP": None,
        "D3BJ": None,
        "Hirshfeld": None,
        "CPCM": "water",
        "RIJCOSX": None,
        "def2/J": None,
        "Grid4": None,
        "GridX4": None,
        "NMR": None,
        "def2/JK": None,
    }

    # Block containing "LOEWDIN ATOMIC CHARGES" has been deleted but is expected by the parser
    logfilename = RESOURCES / "orca/serine-compromised.out"
    with open(logfilename) as f:
        lines = f.readlines()

    properties_dict = orca.read_properties(lines, serine_num_atoms, options)

    # check if there are fewer properties in the dict than expected
    # (should be 5, but LOEWEDIN expected to fail)
    assert len(properties_dict) == 4

    # check if there are still other properties in the dict and the program has not terminated
    assert properties_dict["shielding_constants"][0] == 124.718
    assert properties_dict["hirshfeld_charges"][0] == 0.044416
    assert properties_dict["mulliken_charges"][0] == 0.220691
