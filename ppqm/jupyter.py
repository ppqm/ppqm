"""

Wrapper functions for nglviewer and jupyter helper functions

Reference links
- https://birdlet.github.io/2019/10/02/py3dmol_example/
- http://nglviewer.org/nglview/latest/api.html#nglview.RdkitStructure
- http://nglviewer.org/nglview/latest/api.html#nglview.show_rdkit


Notes on nglviewer usage

>>> import nglview as nv
>>> from rdkit import Chem
... from rdkit.Chem import AllChem
... m = Chem.AddHs(Chem.MolFromSmiles('COc1ccc2[C@H](O)[C@@H](COc2c1)N3CCC(O)(CC3)c4ccc(F)cc4'))
... _ = AllChem.EmbedMultipleConfs(m, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
... view = nv.show_rdkit(m)
... view

>>> # add component m2
>>> # create file-like object
>>> from nglview.show import StringIO
>>> m2 = Chem.AddHs(Chem.MolFromSmiles('N[C@H](C)C(=O)O'))
... fh = StringIO(Chem.MolToPDBBlock(m2))
... view.add_component(fh, ext='pdb')

>>> # load as trajectory, need to have ParmEd
>>> view = nv.show_rdkit(m, parmed=True)

"""

import IPython
import ipywidgets
import nglview
import pandas as pd
from ipywidgets import Layout, interact
from rdkit.Chem import rdMolAlign

from ppqm import chembridge


def show_molobj(molobj, align_conformers=True, show_properties=False):
    """
    Show molobj in jupyter with a slider for each conformer
    """

    if align_conformers:
        rdMolAlign.AlignMolConformers(molobj)

    n_conformers = molobj.GetNumConformers()
    assert n_conformers > 0

    view = nglview.show_rdkit(molobj)

    def _view_conformer(idx):
        coord = chembridge.get_coordinates(molobj, confid=idx)
        view.set_coordinates({0: coord})

        print(f"Conformer {idx} / {n_conformers - 1}")

    if n_conformers > 1:
        interact(
            _view_conformer,
            idx=ipywidgets.IntSlider(min=0, max=n_conformers - 1, step=1),
            layout=Layout(width="100%", height="80px"),
        )

    _view_conformer(0)
    IPython.core.display.display(view)
    if show_properties:
        properties = molobj.GetPropsAsDict()
        pdf = pd.DataFrame([properties]).transpose()
        IPython.core.display.display(pdf)


def show_molobjs(molobjs, align_conformers=True, show_properties=False):
    """ """

    n_molobjs = len(molobjs)

    def _view_molobj(idx):
        show_molobj(
            molobjs[idx], align_conformers=align_conformers, show_properties=show_properties
        )

    interact(
        _view_molobj,
        idx=ipywidgets.IntSlider(min=0, max=n_molobjs - 1, step=1),
        layout=Layout(width="100%", height="80px"),
    )
