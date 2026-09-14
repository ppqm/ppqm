# Psi Phi Quantum Mechanics (ppqm)

Simple bridge between RDKit and quantum chemistry (QC)
packages.

## Installation

```bash
pip install ppqm
```

## Example

All examples assume an RDKit molecule object with hydrogens and a 3D conformer:

```python
from rdkit import Chem
from rdkit.Chem import AllChem

molecule = Chem.MolFromSmiles("O")
molecule = Chem.AddHs(molecule)
AllChem.EmbedMolecule(molecule)
AllChem.UFFOptimizeMolecule(molecule)
```

Create a calculator instance. Here using xTB with 4 cores:

```python
from ppqm import XtbCalculator

xtb = XtbCalculator(cmd="xtb", cores=4)
```

Calculation options are plain dicts, translated into the right input format.
To run a GFN2 optimization in water:

```python
optimize_options = {
    "gfn": 2,
    "alpb": "h2o",
    "opt": None,
}

results = xtb.calculate(molecule, optimize_options)

for i, properties in enumerate(results):
    print(f"Conformer {i} properties: {properties}")
```

More examples in the `notebooks/` directory.

## Supported calculators

- GAMESS
- Gaussian
- MNDO
- MOPAC
- Orca
- xTB

## Related projects

The interface might be too niche, but there are other cool molecular interfaces to explore;

- [ASE](https://gitlab.com/ase/ase)
- [autodE](https://github.com/duartegroup/autodE/)
- [cclib](https://github.com/cclib/cclib)
- [cctk](https://github.com/ekwan/cctk)
- [datamol](https://github.com/datamol-org/datamol)
- [Moleculekit](https://github.com/Acellera/moleculekit)
- [MolSSI](https://github.com/MolSSI)
- [pygamess](https://github.com/kzfm/pygamess)
- [stk](https://github.com/lukasturcani/stk)
- [stko](https://github.com/JelfsMaterialsGroup/stko)

## Future Work

- Would really love to seperate the parsers out and use `cclib` instead.

## Contributing

Fork, branch, and use pre-commit.

## License

MIT — see [LICENSE](LICENSE).
