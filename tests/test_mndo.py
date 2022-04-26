from pathlib import Path

import pytest

from ppqm import chembridge, mndo, tasks

# pytest.skip("Broken module", allow_module_level=True)


def _get_options(scr: Path) -> dict:
    _options = {"scr": scr}
    return _options


def test_optimize_water(tmp_path: Path) -> None:

    mndo_options = _get_options(tmp_path)

    # Get molecule
    smi = "O"
    molobj = chembridge.smiles_to_molobj(smi)
    assert molobj is not None
    molobj = tasks.generate_conformers(molobj, n_conformers=1)

    # Get mndo calculator
    calc = mndo.MndoCalculator(**mndo_options)

    # Optimize water
    options = {
        "pm3": None,
    }

    properties = calc.calculate(molobj, options)
    assert properties is not None

    water_atomization: float = properties["h"]  # type: ignore

    assert pytest.approx(-224.11087077483552, rel=1e-2) == water_atomization
