
Psi Phi Package
===============

Do you need rdkit? Do you need quantum chemistry? We got you bro.


Examples
========

Assume all codesnippets below are using RDKit molecule objs

.. code-block:: python

    mol = Chem.MolFromSmiles("O")
    Chem.AddHydrogens(mol)
    AllChem.UFFOptimizeMolecule(molobj, maxIters=max_steps)

Optimize using MNDO

.. code-block:: python

    from ppqm import MndoCalculator
    calc = MndoCalculator()
    mol = calc.optimize(mol, return_copy=True)

Example of using GAMESS calculator and using specific options

.. code-block:: python

    from ppqm import GamessCalculator
    calc = GamessCalculator(method_options={"method": "pm3"})

    # Overwrite calculation options with GAMESS specific options
    options = dict()
    options["contrl"] = {
        "runtyp": "optimize",
    }
    options["statpt"] = {
        "opttol": 0.005,
        "nstep": 300,
        "projct": False
    }

    # Get properties for each conformer in mol
    properties_list = calc.calculate(mol, options)


different calculation types

.. code-block:: python

    results = calc.properties(molobj)
    results = calc.optimize(molobj)
    results = calc.gradient(molobj)
    results = calc.hessian(molobj)


Other code bases
================

- https://github.com/kzfm/pygamess
- https://github.com/duartegroup/autodE/
- https://github.com/JelfsMaterialsGroup/stko
- https://github.com/lukasturcani/stk
