import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors

from ppqm import chembridge


def generate_conformers(smiles, max_conf=20, min_conf=10, max_steps=1000, random_seed=1):

    molobj = chembridge.smiles_to_molobj(smiles, add_hydrogens=True)

    if molobj is None:
        return None

    status_embed = AllChem.EmbedMolecule(molobj, randomSeed=random_seed)

    if status_embed != 0:
        return None

    status_optim = AllChem.UFFOptimizeMolecule(molobj, maxIters=max_steps)

    # Keep unconverged uff
    if status_optim != 0:
        return None

    # Check bond lengths
    dist = Chem.rdmolops.Get3DDistanceMatrix(molobj)
    np.fill_diagonal(dist, 10.0)
    min_dist = np.min(dist)

    # For some atom_types in UFF, it will fail
    if min_dist < 0.001:
        return None

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
