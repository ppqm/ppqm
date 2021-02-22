import pytest
from context import MNDO_OPTIONS

from ppqm import chembridge, mndo, tasks

pytest.skip("Broken module", allow_module_level=True)


def _get_options(scr):
    _options = {"scr": scr, **MNDO_OPTIONS}
    return _options


def test_optimize_water(tmpdir):

    mndo_options = _get_options(tmpdir)

    # Get molecule
    smi = "O"
    molobj = tasks.generate_conformers(smi, max_conf=1, min_conf=1)

    # Get mndo calculator
    calc = mndo.MndoCalculator(**mndo_options)

    # Optimize water
    properties = calc.optimize(molobj, return_copy=False, return_properties=True)

    water_atomization = properties["h"]

    assert pytest.approx(-224.11087077483552, rel=1e-2) == water_atomization

    return


def test_water_xyz():

    method = "PM3"

    # Get molecule
    molobj = tasks.generate_conformers("O", max_conf=1, min_conf=1)

    # Get XYZ
    atoms, coords, charge = chembridge.molobj_to_axyzc(molobj)

    # Get mndo calculator
    calc = mndo.MndoCalculator(cmd="mndo", scr="_test_dir_", method=method)

    # Optimize coords
    properties = calc.optimize_axyzc(atoms, coords, charge)
    properties = next(properties)

    water_atomization = properties["h"]

    assert pytest.approx(-224.11087077483552, rel=1e-2) == water_atomization

    return
