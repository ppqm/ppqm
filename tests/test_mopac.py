from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import rmsd  # type: ignore[import]
from conftest import RESOURCES

from ppqm import chembridge, mopac, tasks
from ppqm.utils.shell import which

if not which(mopac.MOPAC_CMD):
    pytest.skip("Could not find MOPAC executable", allow_module_level=True)


def _get_options(scr: Path) -> dict:
    mopac_options = {"scr": scr, "cmd": "mopac"}
    return mopac_options


def test_optimize_water_and_get_energy(tmp_path: Path) -> None:

    smi = "O"

    # Get molecule
    n_conformers = 2
    molobj = chembridge.smiles_to_molobj(smi)
    assert molobj is not None
    molobj = tasks.generate_conformers(molobj, n_conformers=n_conformers)

    assert molobj.GetNumAtoms() == 3  # type: ignore[attr-defined]

    mopac_options = _get_options(tmp_path)

    # Get mopac calculator
    calculation_options = {"PM6": None}
    calc = mopac.MopacCalculator(**mopac_options)

    # Optimize water
    properties_per_conformer = calc.calculate(molobj, calculation_options)

    assert len(properties_per_conformer) == n_conformers

    for properties in properties_per_conformer:
        assert properties is not None

        enthalpy_of_formation = properties["h"]  # kcal/mol

        assert pytest.approx(-54.30636, rel=1e-2) == enthalpy_of_formation


def test_multiple_molecules() -> None:

    return


def test_multiple_molecules_with_error(tmp_path: Path) -> None:
    """
    Test for calculating multiple molecules, and error handling for one
    molecule crashing.

    """

    method = "pm6"

    options: Dict[str, Any] = {
        "cmd": "mopac",
        "optimize": True,
        "filename": "mopac_error",
        "scr": tmp_path,
    }

    smis = ["O", "N", "CC", "CC", "CCO"]

    # Change molecule(2) with a weird distance

    # Header
    header = f"{method} mullik precise charge={{charge}} \ntitle {{title}}\n"

    atoms_list: List[List[str]] = []
    coords_list = []
    charge_list = []
    title_list = []

    for i, smi in enumerate(smis):

        molobj = chembridge.smiles_to_molobj(smi)
        assert molobj is not None
        molobj = tasks.generate_conformers(molobj, n_conformers=1)

        atoms, coords, charge = chembridge.get_axyzc(molobj, atomfmt=str)

        if i == 2:
            coords[0, 0] = 0.01
            coords[1, 0] = 0.02

        atoms_list.append(list(atoms))
        coords_list.append(coords)
        charge_list.append(charge)
        title_list.append(f"{smi} {i}")

    properties_list = mopac.properties_from_many_axyzc(
        atoms_list,
        coords_list,
        charge_list,
        header,
        titles=title_list,
        **options,
    )

    for properties in properties_list:
        print(properties)

    return


def test_read_properties() -> None:

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
    properties = result_list[0]
    assert properties is not None
    assert properties.get("h") is not None
    h: float = properties.get("h")  # type: ignore[assignment]
    assert not np.isnan(h)

    # Could not read error molecule idx=2
    assert np.isnan(result_list[error_idx]["h"])  # type: ignore[index]

    return


def test_xyz_usage(tmp_path: Path) -> None:

    mopac_options = _get_options(tmp_path)

    xyz_file = RESOURCES / "compounds/CHEMBL1234757.xyz"

    method = "PM3"

    # Get XYZ
    atoms, coords = rmsd.get_coordinates_xyz(xyz_file)
    charge = 0

    # Header
    title = "test"
    header = f"{method} MULLIK PRECISE charge={{charge}} \nTITLE {title}\n"

    # Optimize coords
    properties = mopac.properties_from_axyzc(atoms, coords, charge, header, **mopac_options)
    assert properties is not None

    # Check energy
    # energy in kcal/mol
    water_atomization = -131.09284
    assert pytest.approx(water_atomization, rel=1e-2) == properties["h"]


def test_options() -> None:
    options: Dict[str, Any] = dict()
    options["pm6"] = None
    options["1scf"] = None
    options["charge"] = 1
    options["title"] = "Test Mol"
    options["precise"] = None

    header = mopac.get_header(options)

    assert type(header) == str
    assert "Test Mol" in header
    assert len(header.split("\n")) == 3
