
Psi Phi Package
===============

Do you need RDKit? Do you need quantum chemistry? We got you.
This package is a simple bridge between RDKit and quantum chemistry (QC) packages
that lack Python interfaces.

Current version has calculator wrappers for

- GAMESS
- Gaussian
- MNDO
- MOPAC
- Orca
- xTB


Example
=======

Assume all codesnippets below are using RDKit molecule objs.

.. code-block:: python

    molecule = Chem.MolFromSmiles("O")
    Chem.AddHydrogens(molecule)
    AllChem.UFFOptimizeMolecule(molecule)


The simple usage is to make an instance of a QC software.
For example, using the popular package xTB, you can define the amount of cores
to allocate and the exact path to the executable.

.. code-block:: python

    from ppqm import XtbCalculator
    xtb = XtbCalculator(cmd="xtb", cores=4)

The format for running calculations are based on Python dictionaries, which are
translated into the right format. So for example running a GFN2 optimization in
water, the input would be

.. code-block:: python

    # Define the calculation
    optimize_options = {,
        "gfn": 2,,
        "alpb": "h2o",
        "opt": None,
    }

    # Run the calculation
    results = xtb.calculate(molecule, optimize_options)

    # Results is a List of Dict properties
    for i, propeties in enumerate(results):
        print(f"Conformer {i} properties: {properties}")

For more documentation by example, checkout the notebooks directory.


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
and find the one right for your project. Know one, not on the list? Add it. In
alphabetic order.

- https://github.com/Acellera/moleculekit
- https://github.com/JelfsMaterialsGroup/stko
- https://github.com/MolSSI
- https://github.com/cclib/cclib
- https://github.com/datamol-org/datamol
- https://github.com/duartegroup/autodE/
- https://github.com/ekwan/cctk
- https://github.com/kzfm/pygamess
- https://github.com/lukasturcani/stk
- https://gitlab.com/ase/ase


Future work
===========

- Separation of concern. The ppqm package should adapt to using `cclib` or
similar to collect quantum output.
