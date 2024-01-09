from pathlib import Path

import pytest

from ppqm import chembridge, mndo, tasks

pytest.skip("Broken interface to MNDO", allow_module_level=True)


def _get_options(scr: Path) -> dict:
    _options = {"scr": scr}
    return _options


def test_optimize_water(tmp_path: Path) -> None:

    mndo_options = _get_options(tmp_path)

    # Get molecule
    smi = "CCCC"
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

    print(properties)

    assert properties is not None
    assert len(properties)

    water_atomization: float = properties[0]["h"]

    assert pytest.approx(-224.11087077483552, rel=1e-2) == water_atomization
