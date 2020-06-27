
import abc


class CalculatorSkeleton(abc.ABC):

    def __init__(self):
        pass

    def get_properties(self, molobj):
        pass


    def optimize(self, molobj):
        pass


    def gradient(self, molobj):
        pass


    def set_optimizer(self, molobj):
        pass


    def set_solvent(self, molobj):
        pass


    def get_solvents(self, molobj):
        pass


    def __repr__(self):
        return "CalculatorSkeleton"

