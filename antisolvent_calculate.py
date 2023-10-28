import numpy as np
from numpy.typing import ArrayLike
import pandas
import matplotlib.pyplot as plt
import ternary

from cosmosac2 import COSMOMolecule, COSMOSAC
from ternary_calculate import TernaryCalculate as tc


class AntisolventCalculate:
    def __init__(self) -> None:
        """
        Class calculating the properties of a ternary drowning-out crystallization system.
        Moreover, validates the antisolvent's effectivenss by calculating the solubility difference.
        """

    def create_system(self) -> tc:
        """
        Creates a system based on the class TernaryCalculate.
        Check the TernaryCalculate script for more info.
        """
        system = tc()
        return system

    def calc_ternary_data(self, system: tc, **kwarg) -> pandas.DataFrame:
        """
        Calculates the ternary data for the given system.

        Paramters
        =========
        system: TernaryCalculate
            The system with solute, solvent, antisolvent.

        Returns
        =======
        DataFrame
            DataFrame with solute, solvent, antisolvent mole fraction.

        Note
        ====
        See TernaryCalculate.calculate for kwarg info.
        """
        ternary_data = system.calculate(
            ngrid=kwarg.get("ngrid", 21), trace=kwarg.get("trace", True)
        )
        ternary_data = pandas.DataFrame(
            ternary_data,
            index=["solute mole frac", "solvent mole frac", "antisolvent mole frac"],
        )

        return ternary_data
