
import pytest
import numpy as np

from context import ppqm

from ppqm import chembridge
from ppqm import gamess
from ppqm import tasks


TMPDIR = "_test_scr_gamess_"

GAMESS_OPTIONS = {
    "scr": TMPDIR,
    "cmd": "rungms",
    "gamess_scr": "~/scr",
    "gamess_userscr": "~/scr",
}


def test_optimization():

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

    header = """ $basis gbasis=pm3 $end
 $contrl runtyp=optimize icharg={charge} $end
 $statpt opttol=0.0005 nstep=300 projct=.F. $end
"""

    molobj = chembridge.sdfstr_to_molobj(methane)
    calc = gamess.GamessCalculator(method="pm3", **GAMESS_OPTIONS)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, header)
    properties = results[0]

    atoms = properties[ppqm.constants.COLUMN_ATOMS]
    energy = properties[ppqm.constants.COLUMN_ENERGY]

    assert (atoms == np.array([6, 1, 1, 1, 1], dtype=int)).all()
    np.testing.assert_almost_equal(energy, -13.0148)

    return


def test_optimization_options():

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
        "basis": {
            "gbasis": "pm3"
        },
        "contrl": {
            "runtyp": "optimize",
            "icharg": "{charge}",
        },
        "statpt": {
            "opttol": 0.0005,
            "nstep": 300,
            "projct": False
        }
    }

    molobj = chembridge.sdfstr_to_molobj(methane)
    calc = gamess.GamessCalculator(**GAMESS_OPTIONS)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, options)
    properties = results[0]

    atoms = properties[ppqm.constants.COLUMN_ATOMS]
    energy = properties[ppqm.constants.COLUMN_ENERGY]

    assert (atoms == np.array([6, 1, 1, 1, 1], dtype=int)).all()
    np.testing.assert_almost_equal(energy, -13.0148)

    return


def test_optimization_read():

    with open("tests/resources/gamess/gamess_methane.log", 'r') as f:
        output = f.readlines()

    properties = gamess.get_properties_coordinates(output)

    atoms = properties[ppqm.constants.COLUMN_ATOMS]
    energy = properties[ppqm.constants.COLUMN_ENERGY]

    assert (atoms == np.array([6, 1, 1, 1, 1], dtype=int)).all()
    np.testing.assert_almost_equal(energy, -13.0148)

    return


def test_vibration():

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

    coordinates = np.array([
        [0., -0., 0., ],
        [-0., -0.88755027, -0.62754422],
        [-0., 0.88755027, -0.62754422],
        [-0.88755027, 0., 0.62754422],
        [0.88755027, 0., 0.62754422],
    ])

    molobj = chembridge.sdfstr_to_molobj(methane)
    chembridge.molobj_set_coordinates(molobj, coordinates)

    header = """
 $basis
     gbasis=PM3
 $end

 $contrl
    scftyp=RHF
    runtyp=hessian
    icharg={charge}
    maxit=60
 $end
"""

    calc = gamess.GamessCalculator(method="pm3", **GAMESS_OPTIONS)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, header)
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
    assert pytest.approx(206.665, rel=3) == properties[
        ppqm.gamess.COLUMN_THERMO][-1, -1]

    assert pytest.approx(-13.01, rel=2) == properties[
        ppqm.constants.COLUMN_ENERGY
    ]

    return


def test_vibration_read():

    with open("tests/resources/gamess/gamess_methane_vib.log", 'r') as f:
        output = f.readlines()

    properties = gamess.get_properties_vibration(output)

    vibs = properties["freq"]
    result = np.array([
        5.757000e+00,
        5.757000e+00,
        9.600000e-02,
        6.419200e+01,
        7.002200e+01,
        7.002200e+01,
        1.362606e+03,
        1.362741e+03,
        1.362741e+03,
        1.451008e+03,
        1.451231e+03,
        3.207758e+03,
        3.207864e+03,
        3.207864e+03,
        3.311312e+03])

    np.testing.assert_almost_equal(vibs, result)

    return


def test_orbitals():

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

    header = """
 $contrl
 coord=cart
 units=angs
 scftyp=rhf
 icharg=0
 maxit=60
 $end
 $basis gbasis=sto ngauss=3 $end
"""

    molobj = chembridge.sdfstr_to_molobj(methane)
    calc = gamess.GamessCalculator(**GAMESS_OPTIONS)

    # calculate returns List(properties) for every conformer
    results = calc.calculate(molobj, header)
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
        0.7505
    ]

    np.testing.assert_almost_equal(orbitals, results)

    return


def test_orbitals_read():

    with open("tests/resources/gamess/gamess_methane_orb.log", 'r') as f:
        output = f.readlines()

    properties = gamess.get_properties_orbitals(output)

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
        0.7505
    ]

    np.testing.assert_almost_equal(orbitals, results)

    return


def test_solvation():

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

    header = """
 $system
    mwords=125
 $end
 $basis
    gbasis=PM3
 $end
 $contrl
    scftyp=RHF
    runtyp=energy
    icharg={:}
 $end
 $pcm
    solvnt=water
    mxts=15000
    icav=1
    idisp=1
 $end
 $tescav
    mthall=4
    ntsall=60
 $end

"""

    calc = gamess.GamessCalculator(**GAMESS_OPTIONS)

    molobj = chembridge.sdfstr_to_molobj(methane)
    results = calc.calculate(molobj, header)
    properties = results[0]

    total_solvation = properties["solvation_total"]
    result = 1.24
    np.testing.assert_almost_equal(total_solvation, result)

    return


def test_solvation_read():

    with open("tests/resources/gamess/gamess_methane_sol.log", 'r') as f:
        output = f.readlines()

    properties = gamess.get_properties_solvation(output)

    total_solvation = properties["solvation_total"]
    result = 1.24
    np.testing.assert_almost_equal(total_solvation, result)

    return


def test_water():

    smi = "O"
    reference_energy = -53.426

    # Get molecule
    n_conformers = 2
    molobj = tasks.generate_conformers(
        smi,
        max_conf=n_conformers,
        min_conf=n_conformers
    )

    # Get mopac calculator
    method = "PM3"
    calc = gamess.GamessCalculator(scr=TMPDIR, method=method)

    results = calc.optimize(molobj, return_properties=True)

    for result in results:
        assert (
            pytest.approx(reference_energy, rel=1e-2)
            ==
            result[ppqm.constants.COLUMN_ENERGY]
        )

    return


def test_get_header():

    options = {
        'contrl': {'scftyp': 'rhf', 'runtyp': 'energy'},
        'basis': {'gbasis': 'sto', 'ngauss': 3},
        'statpt': {'opttol': 0.0001, 'nstep': 20, 'projct': False},
        'system': {'mwords': 30},
    }

    header = ppqm.gamess.get_header(options)
    n_lines = len(header.split("\n"))

    assert n_lines == 4
    assert "runtyp=energy" in header
    assert "nstep=20" in header
    assert "projct=.F." in header


def main():

    # test_output()
    # test_optimization()
    # test_optimization_options()
    # test_vibration_read()
    # test_vibration()
    # test_orbitals_read()
    # test_orbitals()
    # test_solvation_read()
    test_solvation()
    # test_water()
    # test_get_header()

    return


if __name__ == '__main__':
    main()
