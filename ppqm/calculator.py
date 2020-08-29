
import pathlib
import abc

from . import constants

class CalculatorSkeleton(abc.ABC):

    def __init__(self, scr=constants.SCR):

        self.scr = scr

        # Ensure scrdir
        self.set_scratch_directory()

        pass


    def optimize(self, molobj):
        pass


    def gradient(self, molobj):
        pass


    def hessian(self, molobj):
        pass



    def set_optimizer(self, molobj):
        pass


    def set_solvent(self, molobj):
        pass


    def set_energy_unit(self, unit):

        # TODO set unit.convert(value, X, to)

        return


    def set_scratch_directory(self):

        pathlib.Path(self.scr).mkdir(parents=True, exist_ok=True)

        return


    def health_check(self):

        # TODO Check if self.cmd can be found

        return


    def __repr__(self):
        return "CalculatorSkeleton"

