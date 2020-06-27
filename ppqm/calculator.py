
import abc


class CalculatorSkeleton(abc.ABC):

    def __init__(self):

        self.cmd = "ls"
        self.scr = "./"

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

        Path(self.scr).mkdir(parents=True, exist_ok=True)

        return


    def health_check(self):

        # TODO Check if self.cmd can be found

        return


    def __repr__(self):
        return "CalculatorSkeleton"

