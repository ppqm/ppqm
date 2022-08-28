from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from ppqm import chembridge, constants, gamess, tasks
from ppqm.gamess import COLUMN_THERMO, GAMESS_CMD, GamessCalculator
from ppqm.utils.shell import which

if not which(GAMESS_CMD) is None:
    pytest.skip("Could not find GAMESS executable", allow_module_level=True)


def _get_options(scr: Path) -> dict:
    gamess_options = {"scr": scr, "cmd": "rungms"}
    return gamess_options


def test_optimization(tmp_path: Path) -> None:

    gamess_options = _get_options(tmp_path)

    methane = """


  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.8900   -0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8900   -0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8900   -0.0000    0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8900   -0.0000    0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$
    """

    calculation_options = {
        "basis": {"gbasis": "pm3"},
        "contrl": {"runtyp": "optimize"},
        "statpt": {"opttol": 0.0005, "nstep": 300, "projct": False},
    }

    molobj = chembridge.sdfstr_to_molobj(methane)
    assert molobj is not None
    calc = gamess.GamessCalculator(**gamess_options)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, calculation_options)
    properties = results[0]
    print(properties)
    assert properties is not None
    print(properties.keys())

    atoms = properties[constants.COLUMN_ATOMS]
    energy = properties[constants.COLUMN_ENERGY]

    assert (atoms == np.array([6, 1, 1, 1, 1], dtype=int)).all()
    np.testing.assert_almost_equal(energy, -13.0148)

    return


def test_optimization_read() -> None:

    with open("tests/resources/gamess/gamess_methane.log", "r") as f:
        output = f.readlines()

    properties = gamess.get_properties(output)
    assert properties is not None

    atoms = properties[constants.COLUMN_ATOMS]
    energy = properties[constants.COLUMN_ENERGY]

    assert (atoms == np.array([6, 1, 1, 1, 1], dtype=int)).all()
    np.testing.assert_almost_equal(energy, -13.0148)

    assert properties[constants.COLUMN_COORDINATES] is not None


def test_vibration(tmp_path: Path) -> None:

    gamess_options = _get_options(tmp_path)

    methane = """


  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.8900   -0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8900   -0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8900   -0.0000    0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8900   -0.0000    0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$
    """

    coordinates = np.array(
        [
            [
                0.0,
                -0.0,
                0.0,
            ],
            [-0.0, -0.88755027, -0.62754422],
            [-0.0, 0.88755027, -0.62754422],
            [-0.88755027, 0.0, 0.62754422],
            [0.88755027, 0.0, 0.62754422],
        ]
    )

    molobj = chembridge.sdfstr_to_molobj(methane)
    assert molobj is not None
    chembridge.molobj_set_coordinates(molobj, coordinates)

    calculation_options = {
        "basis": {"gbasis": "pm3"},
        "contrl": {"runtyp": "hessian", "maxit": 60},
    }

    molobj = chembridge.sdfstr_to_molobj(methane)
    assert molobj is not None

    calc = GamessCalculator(**gamess_options)
    print(calc)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, calculation_options)
    properties = results[0]
    assert properties is not None

    # GAMESS prints out a thermodynamic table

    #               E         H         G         CV        CP        S
    #            KJ/MOL    KJ/MOL    KJ/MOL   J/MOL-K   J/MOL-K   J/MOL-K
    #  ELEC.      0.000     0.000     0.000     0.000     0.000     0.000
    #  TRANS.     3.718     6.197   -36.542    12.472    20.786   143.348
    #  ROT.       3.718     3.718   -15.045    12.472    12.472    62.932
    #  VIB.     119.279   119.279   119.164     2.252     2.252     0.385
    #  TOTAL    126.716   129.194    67.577    27.195    35.509   206.665
    #  VIB. THERMAL CORRECTION E(T)-E(0) = H(T)-H(0) =        99.870 J/MOL

    assert constants.COLUMN_ENERGY in properties
    assert pytest.approx(206.665, rel=3) == properties[COLUMN_THERMO][-1, -1]
    assert pytest.approx(-13.01, rel=2) == properties[constants.COLUMN_ENERGY]


def test_vibration_read() -> None:

    with open("tests/resources/gamess/gamess_methane_vib.log", "r") as f:
        output = f.readlines()

    properties = gamess.get_properties(output)
    assert properties is not None

    vibs = properties["freq"]
    result = np.array(
        [
            5.757000e00,
            5.757000e00,
            9.600000e-02,
            6.419200e01,
            7.002200e01,
            7.002200e01,
            1.362606e03,
            1.362741e03,
            1.362741e03,
            1.451008e03,
            1.451231e03,
            3.207758e03,
            3.207864e03,
            3.207864e03,
            3.311312e03,
        ]
    )

    np.testing.assert_almost_equal(vibs, result)

    return


def test_orbitals(tmp_path: Path) -> None:

    gamess_options = _get_options(tmp_path)

    methane = """


  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.8900   -0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8900   -0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8900   -0.0000    0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8900   -0.0000    0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$
    """

    options = {
        "contrl": {
            "coord": "cart",
            "units": "angs",
            "scftyp": "rhf",
            "maxit": 60,
        },
        "basis": {"gbasis": "sto", "ngauss": 3},
    }

    molobj = chembridge.sdfstr_to_molobj(methane)
    assert molobj is not None

    calc = gamess.GamessCalculator(**gamess_options)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, options)
    properties = results[0]
    assert properties is not None

    orbitals = properties["orbitals"]
    orbitals_ref: List[float] = [
        -11.0303,
        -0.9085,
        -0.5177,
        -0.5177,
        -0.5177,
        0.713,
        0.713,
        0.713,
        0.7505,
    ]

    np.testing.assert_almost_equal(orbitals, orbitals_ref)


def test_orbitals_read() -> None:

    with open("tests/resources/gamess/gamess_methane_orb.log", "r") as f:
        output = f.readlines()

    properties = gamess.get_properties(output)
    assert properties is not None

    orbitals = properties["orbitals"]
    results = [
        -11.0303,
        -0.9085,
        -0.5177,
        -0.5177,
        -0.5177,
        0.713,
        0.713,
        0.713,
        0.7505,
    ]

    np.testing.assert_almost_equal(orbitals, results)

    return


def test_solvation(tmp_path: Path) -> None:

    gamess_options = _get_options(tmp_path)

    methane = """


  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000   -0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.8900   -0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8900   -0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8900   -0.0000    0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.8900   -0.0000    0.6293 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  1  4  1  0  0  0  0
  1  5  1  0  0  0  0
M  END
$$$$
    """

    molobj = chembridge.sdfstr_to_molobj(methane)
    assert molobj is not None

    options: Dict[str, Any] = dict()
    options["basis"] = {"gbasis": "pm3"}
    options["system"] = {"mwords": 125}
    options["pcm"] = {"solvnt": "water", "mxts": 15000, "icav": 1, "idisp": 1}
    options["tescav"] = {"mthall": 4, "ntsall": 60}

    calc = gamess.GamessCalculator(**gamess_options)

    results = calc.calculate(molobj, options)
    properties = results[0]
    assert properties is not None

    total_solvation = properties["solvation_total"]
    result = 1.24
    np.testing.assert_almost_equal(total_solvation, result)

    return


def test_solvation_read() -> None:

    with open("tests/resources/gamess/gamess_methane_sol.log", "r") as f:
        output = f.readlines()

    properties = gamess.get_properties(output)
    assert properties is not None

    total_solvation = properties["solvation_total"]
    result = 1.24
    np.testing.assert_almost_equal(total_solvation, result)

    return


def test_water(tmp_path: Path) -> None:

    gamess_options = _get_options(tmp_path)

    smi = "O"
    reference_energy = -53.426

    # Get molecule with three conformers
    n_conformers = 3
    molobj = chembridge.smiles_to_molobj(smi)
    assert molobj is not None
    molobj = tasks.generate_conformers(molobj, n_conformers=n_conformers)

    # Get gamess calculator
    calc = gamess.GamessCalculator(**gamess_options)

    options = {
        "basis": {"gbasis": "pm3"},
        "contrl": {"runtyp": "optimize"},
        "statpt": {"opttol": 0.0005, "nstep": 300, "projct": False},
    }

    results = calc.calculate(molobj, options)

    for result in results:
        assert result is not None
        assert pytest.approx(reference_energy, rel=1e-2) == result[constants.COLUMN_ENERGY]


def test_fail_wrong_method(tmp_path: Path) -> None:

    gamess_options = _get_options(tmp_path)

    # Get molecule with three conformers
    smi = "O"
    molobj = chembridge.smiles_to_molobj(smi)
    assert molobj is not None
    molobj = tasks.generate_conformers(molobj, n_conformers=1)

    # Set bad options
    options = {
        "basis": {"gbasis": "pm9000"},
        "contrl": {
            "runtyp": "optimize",
        },
        "statpt": {"opttol": 0.0005, "nstep": 300, "projct": False},
    }

    # Get gamess calculator and calculate molobj with bad methods
    calc = gamess.GamessCalculator(**gamess_options)
    results = calc.calculate(molobj, options)

    # will return None for each failed conformer

    assert results is not None
    assert isinstance(results, list)
    assert results[0] is None


def test_get_header() -> None:

    options = {
        "contrl": {"scftyp": "rhf", "runtyp": "energy"},
        "basis": {"gbasis": "sto", "ngauss": 3},
        "statpt": {"opttol": 0.0001, "nstep": 20, "projct": False},
        "system": {"mwords": 30},
    }

    header = gamess.get_header(options)
    n_lines = len(header.split("\n"))

    assert n_lines == 4
    assert "runtyp=energy" in header
    assert "nstep=20" in header
    assert "opttol=0.0001" in header
    assert "projct=.F." in header


def test_type() -> None:

    # TODO read properties from log files

    return


def test_dinitrogen(tmp_path: Path) -> None:

    sdf = """


  2  1  0  0  0  0  0  0  0  0999 V2000
    0.7500    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0
M  END """

    gamess_options = _get_options(tmp_path)

    molobj = chembridge.sdfstr_to_molobj(sdf)
    assert molobj is not None
    calc = gamess.GamessCalculator(**gamess_options)

    print(calc)
    options = {
        "basis": {"gbasis": "pm3"},
        "contrl": {"runtyp": "optimize"},
        "statpt": {"opttol": 0.0005, "nstep": 300, "projct": False},
    }

    results = calc.calculate(molobj, options)

    assert isinstance(results, list)

    properties = results[0]
    assert isinstance(properties, dict)
    assert pytest.approx(17.56621, rel=10**-4) == properties[constants.COLUMN_ENERGY]
