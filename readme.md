# Psi Phi Package

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

## Example

Assume all codesnippets below are using RDKit molecule objs.

```python
molecule = Chem.MolFromSmiles("O")
Chem.AddHydrogens(molecule)
AllChem.UFFOptimizeMolecule(molecule)
```

The simple usage is to make an instance of a QC software.
For example, using the popular package xTB, you can define the amount of cores
to allocate and the exact path to the executable.

```python
from ppqm import XtbCalculator

xtb = XtbCalculator(cmd="xtb", cores=4)
```

The format for running calculations are based on Python dictionaries, which are
translated into the right format. So for example running a GFN2 optimization in
water, the input would be

```python
# Define the calculation
optimize_options = {
    "gfn": 2,
    "alpb": "h2o",
    "opt": None,
}

# Run the calculation
results = xtb.calculate(molecule, optimize_options)

# Results is a List of Dict properties
for i, propeties in enumerate(results):
    print(f"Conformer {i} properties: {properties}")
```

For more documentation by example, checkout the `notebooks/` directory.

## Contributions

Fork, branch and use pre-commit.

## Other code bases

"is this the first python wrapper for quantum chemistry?" No, check the others
and find the one right for your project. Know one, not on the list? Add it. In
alphabetic order.

- [Moleculekit](https://github.com/Acellera/moleculekit)
- [stko](https://github.com/JelfsMaterialsGroup/stko)
- [MolSSI](https://github.com/MolSSI)
- [cclib](https://github.com/cclib/cclib)
- [datamol](https://github.com/datamol-org/datamol)
- [autodE](https://github.com/duartegroup/autodE/)
- [cctk](https://github.com/ekwan/cctk)
- [pygamess](https://github.com/kzfm/pygamess)
- [stk](https://github.com/lukasturcani/stk)
- [ASE](https://gitlab.com/ase/ase)

## Future work

- Separation of concern. The ppqm package should adapt to using `cclib` or similar to collect quantum output.
