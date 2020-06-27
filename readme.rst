
Psi Phi Package
===============

Do you need rdkit? Do you need quantum chemistry? We got you bro.


Examples
========

Optimize water

.. code-block:: python

    from ppqm import MndoCalculator
    from rdkit import Chem
    calc = MndoCalculator()
    mol = Chem.MolFromSmiles("O")
    Chem.AddHydrogens(mol)
    AllChem.UFFOptimizeMolecule(molobj, maxIters=max_steps)
    mol = calc.optimize(mol)


