import numpy as np
import pytest
import rmsd
from context import MOPAC_OPTIONS, RESOURCES

from ppqm import chembridge, mopac, tasks


def _get_options(scr):
    mopac_options = {"scr": scr, **MOPAC_OPTIONS}
    return mopac_options


def test_optimize_water_and_get_energy(tmpdir):

    smi = "O"

    # Get molecule
    n_conformers = 2
    molobj = tasks.generate_conformers(smi, n_conformers=n_conformers)

    assert molobj.GetNumAtoms() == 3

    mopac_options = _get_options(tmpdir)

    # Get mopac calculator
    calculation_options = {"PM6": None}
    calc = mopac.MopacCalculator(**mopac_options)

    # Optimize water
    properties_per_conformer = calc.optimize(
        molobj, options=calculation_options, return_copy=False, return_properties=True
    )

    assert len(properties_per_conformer) == n_conformers

    for properties in properties_per_conformer:

        enthalpy_of_formation = properties["h"]  # kcal/mol

        assert pytest.approx(-54.30636, rel=1e-2) == enthalpy_of_formation


def test_multiple_molecules():

    return


def test_multiple_molecules_with_error(tmpdir):
    """
    Test for calculating multiple molecules, and error handling for one
    molecule crashing.

    """

    method = "pm6"

    options = {
        "cmd": "mopac",
        "optimize": True,
        "filename": "mopac_error",
        "scr": tmpdir,
    }

    smis = ["O", "N", "CC", "CC", "CCO"]

    # Change molecule(2) with a weird distance

    # Header
    header = f"{method} mullik precise charge={{charge}} \ntitle {{title}}\n"

    atoms_list = []
    coords_list = []
    charge_list = []
    title_list = []

    for i, smi in enumerate(smis):

        molobj = tasks.generate_conformers(smi, n_conformers=1)

        atoms, coords, charge = chembridge.get_axyzc(molobj, atomfmt=str)

        if i == 2:
            coords[0, 0] = 0.01
            coords[1, 0] = 0.02

        atoms_list.append(atoms)
        coords_list.append(coords)
        charge_list.append(charge)
        title_list.append(f"{smi} {i}")

    properties_list = mopac.properties_from_many_axyzc(
        atoms_list,
        coords_list,
        charge_list,
        header,
        titles=title_list,
        **options,
    )

    for properties in properties_list:
        print(properties)

    return


def test_read_properties():

    filename = "tests/resources/mopac/output_with_error.txt"
    error_idx = 2

    output = mopac.read_output(filename, translate_filename=False)

    result_list = []
    for lines in output:
        properties = mopac.get_properties(lines)
        result_list.append(properties)

    # Could read all properties
    assert len(result_list) == 5

    # Could read correct results
    assert result_list[0]["h"] is not None
    assert not np.isnan(result_list[0]["h"])

    # Could not read error molecule idx=2
    assert np.isnan(result_list[error_idx]["h"])

    return


def test_xyz_usage(tmpdir):

    mopac_options = _get_options(tmpdir)

    xyz_file = RESOURCES / "compounds/CHEMBL1234757.xyz"

    method = "PM3"

    # Get XYZ
    atoms, coords = rmsd.get_coordinates_xyz(xyz_file)
    charge = 0

    # Header
    title = "test"
    header = f"{method} MULLIK PRECISE charge={{charge}} \nTITLE {title}\n"

    # Optimize coords
    properties = mopac.properties_from_axyzc(atoms, coords, charge, header, **mopac_options)

    # Check energy
    # energy in kcal/mol
    water_atomization = -131.09284
    assert pytest.approx(water_atomization, rel=1e-2) == properties["h"]


def test_options():
    options = dict()
    options["pm6"] = None
    options["1scf"] = None
    options["charge"] = 1
    options["title"] = "Test Mol"
    options["precise"] = None

    header = mopac.get_header(options)

    assert type(header) == str
    assert "Test Mol" in header
    assert len(header.split("\n")) == 3
