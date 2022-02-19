import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
from rdkit.Chem import rdDistGeom

from ppqm import chembridge


def generate_conformers_legacy(molobj, max_conf=20, min_conf=10, random_seed=1, return_copy=True):

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
    molecule, n_conformers=None, max_conformers=500, return_copy=True, seed=0xF00D
):
    """ Generate 3D conformers using RDKit ETKDGv3 """

    if isinstance(molecule, str):
        # assume smiles
        molecule = chembridge.smiles_to_molobj(molecule)

    if return_copy:
        molobj = chembridge.copy_molobj(molecule)
    else:
        molobj = molecule

    embed_parameters = rdDistGeom.ETKDGv3()
    embed_parameters.randomSeed = seed

    if n_conformers is None:
        rot_bond = rdMolDescriptors.CalcNumRotatableBonds(molobj)
        n_conformers = 1 + 3 * rot_bond

    n_conformers = min(n_conformers, max_conformers)
    molobj = Chem.AddHs(molobj)
    rdDistGeom.EmbedMultipleConfs(molobj, n_conformers, embed_parameters)

    return molobj


def optimize_molobj_uff(molobj, max_steps=1000):
    """ Optimize molobj with UFF """

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
