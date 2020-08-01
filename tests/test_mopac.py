
import pytest

from context import ppqm

from ppqm import mopac, tasks

# TODO use tempfile

def test_optimize_water_and_get_energy():

    smi = "O"

    # Get molecule
    n_conformers = 2
    molobj = tasks.generate_conformers("O", max_conf=n_conformers, min_conf=n_conformers)

    # Get mndo calculator
    method = "PM6"

    calc = mopac.MopacCalculator(scr="_test_dir_", method=method)

    # Optimize water
    properties_per_conformer = calc.optimize(
        molobj,
        return_copy=False,
        return_properties=True)

    assert len(properties_per_conformer) == n_conformers

    for properties in properties_per_conformer:

        enthalpy_of_formation = properties["h"]
        assert pytest.approx(-54.30636, rel=1e-2) == enthalpy_of_formation # kcal/mol

    return


def test_read_properties():
    # TODO
    return


def test_func_usage():
    # TODO Define ATOMS, XYZ, CHARGE examples
    return

