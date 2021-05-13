import numpy as np
import pytest
from context import GAMESS_OPTIONS

import ppqm
from ppqm import chembridge, gamess, tasks
from ppqm.gamess import GamessCalculator


def _get_options(scr):
    gamess_options = {"scr": scr, **GAMESS_OPTIONS}
    return gamess_options


def test_optimization(tmpdir):

    gamess_options = _get_options(tmpdir)

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
        "contrl": {"runtyp": "optimize"},
        "statpt": {"opttol": 0.0005, "nstep": 300, "projct": False},
    }

    molobj = chembridge.sdfstr_to_molobj(methane)
    calc = gamess.GamessCalculator(method_options={"method": "pm3"}, **gamess_options)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, calculation_options)
    properties = results[0]

    atoms = properties[ppqm.constants.COLUMN_ATOMS]
    energy = properties[ppqm.constants.COLUMN_ENERGY]

    assert (atoms == np.array([6, 1, 1, 1, 1], dtype=int)).all()
    np.testing.assert_almost_equal(energy, -13.0148)

    return


def test_optimization_read():

    with open("tests/resources/gamess/gamess_methane.log", "r") as f:
        output = f.readlines()

    properties = gamess.get_properties(output)

    atoms = properties[ppqm.constants.COLUMN_ATOMS]
    energy = properties[ppqm.constants.COLUMN_ENERGY]

    assert (atoms == np.array([6, 1, 1, 1, 1], dtype=int)).all()
    np.testing.assert_almost_equal(energy, -13.0148)

    assert properties[ppqm.constants.COLUMN_COORDINATES] is not None


def test_vibration(tmpdir):

    gamess_options = _get_options(tmpdir)

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
    chembridge.molobj_set_coordinates(molobj, coordinates)

    method_options = {"method": "pm3"}
    calculation_options = {
        "contrl": {"runtyp": "hessian", "maxit": 60},
    }

    molobj = chembridge.sdfstr_to_molobj(methane)
    calc = GamessCalculator(method_options=method_options, **gamess_options)
    print(calc)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, calculation_options)
    properties = results[0]

    # GAMESS prints out a thermodynamic table

    #               E         H         G         CV        CP        S
    #            KJ/MOL    KJ/MOL    KJ/MOL   J/MOL-K   J/MOL-K   J/MOL-K
    #  ELEC.      0.000     0.000     0.000     0.000     0.000     0.000
    #  TRANS.     3.718     6.197   -36.542    12.472    20.786   143.348
    #  ROT.       3.718     3.718   -15.045    12.472    12.472    62.932
    #  VIB.     119.279   119.279   119.164     2.252     2.252     0.385
    #  TOTAL    126.716   129.194    67.577    27.195    35.509   206.665
    #  VIB. THERMAL CORRECTION E(T)-E(0) = H(T)-H(0) =        99.870 J/MOL

    assert ppqm.constants.COLUMN_ENERGY in properties
    assert pytest.approx(206.665, rel=3) == properties[ppqm.gamess.COLUMN_THERMO][-1, -1]

    assert pytest.approx(-13.01, rel=2) == properties[ppqm.constants.COLUMN_ENERGY]

    return


def test_vibration_read():

    with open("tests/resources/gamess/gamess_methane_vib.log", "r") as f:
        output = f.readlines()

    properties = gamess.get_properties(output)

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


def test_orbitals(tmpdir):

    gamess_options = _get_options(tmpdir)

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
    calc = gamess.GamessCalculator(**gamess_options)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, options)
    properties = results[0]

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


def test_orbitals_read():

    with open("tests/resources/gamess/gamess_methane_orb.log", "r") as f:
        output = f.readlines()

    properties = gamess.get_properties(output)

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


def test_solvation(tmpdir):

    gamess_options = _get_options(tmpdir)

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

    options = dict()
    options["system"] = {"mwords": 125}
    options["pcm"] = {"solvnt": "water", "mxts": 15000, "icav": 1, "idisp": 1}
    options["tescav"] = {"mthall": 4, "ntsall": 60}

    calc = gamess.GamessCalculator(method_options={"method": "pm3"}, **gamess_options)

    results = calc.calculate(molobj, options)
    properties = results[0]

    total_solvation = properties["solvation_total"]
    result = 1.24
    np.testing.assert_almost_equal(total_solvation, result)

    return


def test_solvation_read():

    with open("tests/resources/gamess/gamess_methane_sol.log", "r") as f:
        output = f.readlines()

    properties = gamess.get_properties(output)

    total_solvation = properties["solvation_total"]
    result = 1.24
    np.testing.assert_almost_equal(total_solvation, result)

    return


def test_water(tmpdir):

    gamess_options = _get_options(tmpdir)

    smi = "O"
    reference_energy = -53.426

    # Get molecule with three conformers
    n_conformers = 3
    molobj = tasks.generate_conformers(smi, n_conformers=n_conformers)

    # Get gamess calculator
    method = "PM3"
    method_options = {"method": method}
    calc = gamess.GamessCalculator(method_options=method_options, **gamess_options)

    results = calc.optimize(molobj, return_properties=True)

    for result in results:
        assert pytest.approx(reference_energy, rel=1e-2) == result[ppqm.constants.COLUMN_ENERGY]

    return


def test_fail_wrong_method(tmpdir):

    gamess_options = _get_options(tmpdir)

    # Get molecule with three conformers
    smi = "O"
    molobj = tasks.generate_conformers(smi, n_conformers=1)

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
    assert isinstance(results[0], dict)
    assert "error" in results[0]


def test_get_header():

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


def test_type():

    # TODO read properties from log files

    return


def test_dinitrogen(tmpdir):

    sdf = """


  2  1  0  0  0  0  0  0  0  0999 V2000
    0.7500    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  3  0
M  END """

    gamess_options = _get_options(tmpdir)

    molobj = chembridge.sdfstr_to_molobj(sdf)
    calc = gamess.GamessCalculator(**gamess_options)
    results = calc.optimize(molobj, return_properties=True)

    assert isinstance(results, list)

    properties = results[0]
    assert isinstance(properties, dict)
    assert pytest.approx(17.56621, rel=10 ** -4) == properties[ppqm.constants.COLUMN_ENERGY]
