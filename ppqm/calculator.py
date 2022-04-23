import abc
import copy
import logging
from collections import ChainMap
from pathlib import Path
from typing import Any, List, Optional

from ppqm import chembridge, constants
from ppqm.chembridge import Mol

_logger = logging.getLogger(__name__)


class BaseCalculator(abc.ABC):
    """Base class for quantum calculators

    This class should not be used directly, use a class appropriate for your
    quantum calculations (e.g. MopacCalculator or GamessCalculator) instead.
    """

    def __init__(self, scr: Path = constants.SCR) -> None:
        self.scr = Path(scr)
        # Ensure scrdir
        self.set_scratch_directory()

    def _health_check(self) -> None:
        raise NotImplementedError

    def _generate_options(self, **kwargs: Any) -> dict:
        """to be implemented by individual programs"""
        raise NotImplementedError

    def calculate(self, molobj: Mol, options: dict) -> List[Optional[dict]]:
        raise NotImplementedError

    def optimize(self, molobj: Mol, options: dict = {}, return_copy: bool = True) -> Mol:
        """

        Parameters
        ----------
        molobj: Mol
            A RDkit Molobj

        return_copy: Bool
            Return a new copy of molobj, instead of overwriting it

        return_properties: Bool
            Return list of properties for molobj conformers

        Examples
        --------
        >>> molobj_prime = calc.optimize(molobj)

        Returns
        -------
        molobj: Mol
            RDKit molobj with updated conformer coordinates

        properties: List(Dict(Str, Any))
            Properties associated with each conformer

        """

        # TODO Embed properties into conformres

        # Merge options
        options_ = self._generate_options(optimize=True)
        options_prime = dict(ChainMap(options, options_))

        if return_copy:
            molobj = copy.deepcopy(molobj)

        result_properties: List[dict] = self.calculate(molobj, options_prime)  # type: ignore

        for i, properties in enumerate(result_properties):

            # TODO Check if unconverged
            # TODO Check number of steps?

            if constants.COLUMN_COORDINATES not in properties:
                # TODO Unable to set coordinates, skip for now
                _logger.error(f"Unable to optimize, conformer {i} skipped")
                continue

            coord = properties[constants.COLUMN_COORDINATES]

            # Set coord on conformer
            chembridge.molobj_set_coordinates(molobj, coord, confid=i)

        return molobj

    def get_gradient(self, molobj: Mol) -> None:
        raise NotImplementedError

    def get_hessian(self, molobj: Mol) -> None:
        raise NotImplementedError

    def set_energy_unit(self, unit: Mol) -> None:
        raise NotImplementedError
        # TODO set unit.convert(value, X, to)

    def set_scratch_directory(self) -> None:
        self.scr.mkdir(parents=True, exist_ok=True)

    def health_check(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "CalculatorSkeleton()"
