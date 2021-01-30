
Psi Phi Package
===============

Do you need rdkit? Do you need quantum chemistry? We got you bro.


Examples
========

Assume all codesnippets below are using RDKit molecule objs

.. code-block:: python

    molecule = Chem.MolFromSmiles("O")
    Chem.AddHydrogens(mol)
    AllChem.UFFOptimizeMolecule(molobj)

Optimize using XTB

.. code-block:: python

    from ppqm.xtblib import XtbCalculator
    xtb = XtbCalculator()
    molecule2 = xtb.optimize(molecule, return_copy=True)

Example of using GAMESS calculator and using specific options.
As you notice, GAMESS needs a lot of settings to work with.

.. code-block:: python

    from ppqm import GamessCalculator

    # Let's set the complicated GAMESS settings
    gamess_options = {
        "scr": "/tmp/node/scr/space/slurm/id",
        "cmd": "/opt/gamess/rungms",
        "gamess_scr": "~/scr",
        "gamess_userscr": "~/userscr",
    }
    gamess = GamessCalculator(**gamess_options)

    # Now that we have gamess setup, we can then the GAMESS options we all know
    # and love. Knowing exactly what keywords to set in GAMESS, you'll have to
    # read the manual
    calculation_option = {
        "runtyp": "optimize",
        "statpt": {
            "opttol": 0.005,
            "nstep": 300,
            "projct": False,
        }
    }

    # We then use the options to get properties for the molecule.
    # The return will be a list of dictionaries, per conformer in the molobj.
    properties_list = gamess.calculate(molecule, calculation_option)


Contributions
=============

Fork, branch and use pre-commit.



Other code bases
================

"is this the first python wrapper for quantum chemistry?" No, check the others
and find the one right for your project. Know one, not on the list? Add it.


- https://github.com/kzfm/pygamess
- https://github.com/duartegroup/autodE/
- https://github.com/JelfsMaterialsGroup/stko
- https://github.com/lukasturcani/stk
