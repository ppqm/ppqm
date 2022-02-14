import abc
import copy
import logging
import pathlib
from collections import ChainMap

from . import chembridge, constants

_logger = logging.getLogger(__name__)


class BaseCalculator(abc.ABC):
    """Base class for quantum calculators

    This class should not be used directly, use a class appropriate for your
    quantum calculations (e.g. MopacCalculator or GamessCalculator) instead.
    """

    def __init__(self, scr=constants.SCR):

        self.scr = pathlib.Path(scr)

        # Ensure scrdir
        self.set_scratch_directory()

        return

    def _health_check(self):
        return

    def _generate_options(self, **kwargs):
        """ to be implemented by individual programs """
        return

    def properties(
        self,
        molobj,
    ):
        """

        Parameters
        ----------
        molobj: Mol
            A RDkit Molobj

        Examples
        --------
        >>> # Get properties per conformer
        >>> results = calc.properties(molobj)
        >>> for properties in results:
        >>>     print(properties)

        Returns
        -------
        properties_list: List(Dict(Str, Any))
            List of properties per conformer in associated with RDKit Mol

        """

        header = self._generate_options(optimize=False)

        results = self.calculate(molobj, header)

        return results

    def optimize(self, molobj, options={}, return_copy=True, return_properties=False):
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

        # Merge options
        options_ = self._generate_options(optimize=True)
        options_prime = ChainMap(options, options_)
        options_prime = dict(options_prime)

        if return_copy:
            molobj = copy.deepcopy(molobj)

        result_properties = self.calculate(molobj, options_prime)

        if return_properties:
            return list(result_properties)

        for i, properties in enumerate(result_properties):

            # TODO Check if unconverged
            # TODO Check number of steps?

            if constants.COLUMN_COORDINATES not in properties:
                # TODO Unable to set coordinates, skip for now
                _logger.error(f"Unable to optimize, conformer {i} skipped")
                continue

            coord = properties[constants.COLUMN_COORDINATES]

            # Set coord on conformer
            chembridge.molobj_set_coordinates(molobj, coord, idx=i)

        return molobj

    def get_gradient(self, molobj):
        pass

    def get_hessian(self, molobj):
        pass

    def set_optimizer(self, molobj):
        pass

    def set_solvent(self, molobj):
        pass

    def set_energy_unit(self, unit):

        # TODO set unit.convert(value, X, to)

        return

    def set_scratch_directory(self):

        self.scr.mkdir(parents=True, exist_ok=True)

        return

    def clean_scratch_directory(self):

        return

    def health_check(self):

        # TODO Check if self.cmd can be found

        return

    def __repr__(self):
        return "CalculatorSkeleton()"
