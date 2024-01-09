""" Collection of common cheminfo tasks """

from typing import Optional

import rdkit.Chem as Chem  # type: ignore[import-untyped]
import rdkit.Chem.AllChem as AllChem  # type: ignore[import-untyped]
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors  # type: ignore[import-untyped]
from rdkit.Chem import rdDistGeom

from ppqm import chembridge
from ppqm.chembridge import Mol


def generate_conformers_legacy(
    molobj: Mol,
    max_conf: int = 20,
    min_conf: int = 10,
    return_copy: int = True,
) -> Mol:
    """

    Args:
        molobj (Mol):
        max_conf (int):
        min_conf (int):
        random_seed (int):
        return_copy (int):

    Returns:
        Mol
    """
    if return_copy:
        molobj = chembridge.copy_molobj(molobj)

    rot_bond = rdMolDescriptors.CalcNumRotatableBonds(molobj)

    confs = min(1 + 3 * rot_bond, max_conf)
    confs = max(confs, min_conf)

    _ = AllChem.EmbedMultipleConfs(
        molobj,
        numConfs=confs,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
    )

    return molobj


def generate_conformers(
    molobj: Mol,
    n_conformers: Optional[int] = None,
    max_conformers: int = 500,
    return_copy: bool = True,
    random_seed: int = 61453,
) -> Mol:
    """Generate 3D conformers using RDKit ETKDGv3"""

    if return_copy:
        molobj = chembridge.copy_molobj(molobj)

    embed_parameters = rdDistGeom.ETKDGv3()
    embed_parameters.randomSeed = random_seed

    if n_conformers is None:
        rot_bond = rdMolDescriptors.CalcNumRotatableBonds(molobj)
        n_conformers = 1 + 3 * rot_bond

    assert n_conformers is not None

    n_conformers = min(n_conformers, max_conformers)
    molobj = Chem.AddHs(molobj)
    rdDistGeom.EmbedMultipleConfs(molobj, n_conformers, embed_parameters)

    return molobj


def optimize_molobj_uff(molobj: Mol, max_steps: int = 1000) -> Optional[Mol]:
    """Optimize molobj with UFF"""

    status_embed = AllChem.EmbedMolecule(molobj)

    if status_embed != 0:
        return None

    try:
        status_optim = AllChem.UFFOptimizeMolecule(molobj, maxIters=max_steps)
    except RuntimeError:
        return None

    # Don't keep unconverged uff
    if status_optim != 0:
        return None

    # Check min bond lengths
    status_structure = chembridge.molobj_check_distances(molobj, max_cutoff=None)

    if max(status_structure) != 0:
        return None

    return molobj
