
Psi Phi Package
===============

Do you need rdkit? Do you need quantum chemistry? We got you bro.


Examples
========

Assume all codesnippets below are using RDKit molecule objs

.. code-block:: python

    molecule = Chem.MolFromSmiles("O")
    Chem.AddHydrogens(molecule)
    AllChem.UFFOptimizeMolecule(molecule)

Optimize using XTB

.. code-block:: python

    from ppqm import XtbCalculator
    xc = XtbCalculator()
    molecule2 = xc.optimize(molecule, return_copy=True)

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
    gc = GamessCalculator(**gamess_options)

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
    results = gc.calculate(molecule, calculation_option)

    for properties in results:
        print(properties)


Notes on Jupyter usage
======================

Example notebooks uses nglview to visualize the molecules in notebooks


.. code-block:: bash

    # install nglview
    conda install nglview -c conda-forge
    # install plugin for jupyter lab
    jupyter labextension install nglview-js-widgets

Please note, if you are using Jupyter Lab (not notebook) there are som
additional notes to consider


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
- https://github.com/MolSSI
- https://github.com/datamol-org/datamol
- https://github.com/ekwan/cctk
