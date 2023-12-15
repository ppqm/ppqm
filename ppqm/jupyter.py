from typing import List

import IPython
import ipywidgets  # type: ignore[import-untyped]
import nglview  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from ipywidgets import Layout, interact
from rdkit.Chem import rdMolAlign  # type: ignore[import-untyped]

from ppqm import chembridge
from ppqm.chembridge import Mol


def show_molobj(molobj: Mol, align_conformers: bool = True, show_properties: bool = False) -> None:
    """Show molobj in jupyter with a slider for each conformer"""

    if align_conformers:
        rdMolAlign.AlignMolConformers(molobj)

    n_conformers = molobj.GetNumConformers()
    assert n_conformers > 0

    view = nglview.show_rdkit(molobj)

    def _view_conformer(idx: int) -> None:
        coord = chembridge.get_coordinates(molobj, confid=idx)
        view.set_coordinates({0: coord})

    if n_conformers > 1:
        interact(
            _view_conformer,
            idx=ipywidgets.IntSlider(min=0, max=n_conformers - 1, step=1),
            layout=Layout(width="100%", height="80px"),
        )

    else:
        _view_conformer(0)

    IPython.core.display.display(view)
    if show_properties:
        properties: dict = molobj.GetPropsAsDict()  # type: ignore
        pdf = pd.DataFrame([properties]).transpose()
        IPython.core.display.display(pdf)


def show_molobjs(
    molobjs: List[Mol], align_conformers: bool = True, show_properties: bool = False
) -> None:
    """ """

    n_molobjs = len(molobjs)

    def _view_molobj(idx: int) -> None:
        show_molobj(
            molobjs[idx], align_conformers=align_conformers, show_properties=show_properties
        )

    interact(
        _view_molobj,
        idx=ipywidgets.IntSlider(min=0, max=n_molobjs - 1, step=1),
        layout=Layout(width="100%", height="80px"),
    )
