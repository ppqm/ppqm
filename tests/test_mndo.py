
from context import ppqm

from ppqm import mndo, tasks, chembridge

# TODO Use config for commands
# TODO Use tempfile with constructor


def test_optimize_water():

    smi = "O"

    # Use tmpdir
    # Path(scrdir).mkdir(parents=True, exist_ok=True)

    # Get molecule
    molobj = tasks.generate_conformers("O", max_conf=1, min_conf=1)

    # Get mndo calculator
    calc = mndo.MndoCalculator(cmd="mndo", scr="_test_dir_")

    method = "PM3"

    # Optimize water
    properties = calc.optimize(molobj,
        return_copy=False,
        return_properties=True)

    water_atomization = properties["h"]

    assert pytest.approx(-224.11087077483552, rel=1e-2) == water_atomization

    return


def test_water_xyz():

    smi = "O"
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




