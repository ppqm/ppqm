
import numpy as np
import pytest

from context import ppqm
from ppqm import mopac, tasks, chembridge

# TODO use tempfile

TMPDIR = "_test_scr_mopac_"


def test_optimize_water_and_get_energy():

    smi = "O"

    # Get molecule
    n_conformers = 2
    molobj = tasks.generate_conformers(
        smi,
        max_conf=n_conformers,
        min_conf=n_conformers
    )

    # Get mopac calculator
    method = "PM6"
    calc = mopac.MopacCalculator(scr=TMPDIR, method=method)

    # Optimize water
    properties_per_conformer = calc.optimize(
        molobj,
        return_copy=False,
        return_properties=True)

    assert len(properties_per_conformer) == n_conformers

    for properties in properties_per_conformer:

        enthalpy_of_formation = properties["h"]
        # kcal/mol
        assert pytest.approx(-54.30636, rel=1e-2) == enthalpy_of_formation

    return


def test_multiple_molecules():

    return


def test_multiple_molecules_with_error():

    method = "pm6"

    options = {
        "cmd": "mopac",
        "optimize": True,
        "filename": "mopac_error",
        "scr": TMPDIR,
        "debug": True
    }

    smis = ["O", "N", "CC", "CC", "CCO"]

    # Change molecule(2) with a weird distance

    # Header
    title = "test"
    header = f"{method} mullik precise charge={{charge}} \ntitle {{title}}\n"

    atoms_list = []
    coords_list = []
    charge_list = []
    title_list = []

    for i, smi in enumerate(smis):
        molobj = tasks.generate_conformers(smi, max_conf=1, min_conf=1)
        atoms, coords, charge = chembridge.molobj_to_axyzc(molobj,
            atom_type=str)

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
        **options)

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
    assert result_list[0]['h'] is not None
    assert not np.isnan(result_list[0]['h'])

    # Could not read error molecule
    assert np.isnan(result_list[error_idx]['h'])

    return


def test_xyz_usage():

    smi = "O"
    method = "PM3"
    options = {"scr": TMPDIR}

    # Get molecule
    molobj = tasks.generate_conformers(smi, max_conf=1, min_conf=1)

    # Get XYZ
    atoms, coords, charge = chembridge.molobj_to_axyzc(molobj, atom_type=str)

    # Header
    title = "test"
    header = f"{method} MULLIK PRECISE charge={{charge}} \nTITLE {title}\n"

    # Optimize coords
    properties = mopac.properties_from_axyzc(atoms, coords, charge, header, **options)

    # Check energy
    # energy in kcal/mol
    water_atomization = -50.88394
    assert pytest.approx(water_atomization, rel=1e-2) == properties["h"]

    return
