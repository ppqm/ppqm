
from context import ppqm

from ppqm import mndo


def test_optimize_water():

    smi = "O"

    # Use tmpdir
    # Path(scrdir).mkdir(parents=True, exist_ok=True)

    # Get molecule
    molobj = cheminfo.generate_conformers("O", max_conf=1, min_conf=1)

    # Get mndo calculator
    calc = mndo.MndoCalculator(cmd="mndo", scr="_test_dir_")

    method = "PM3"

    # Optimize water
    properties = calc.optimize(molobj,
        method,
        return_copy=False,
        return_propeties=True)

    water_atomization = properties["h"]

    assert pytest.approx(-224.11087077483552, rel=1e-2) == water_atomization

    return


