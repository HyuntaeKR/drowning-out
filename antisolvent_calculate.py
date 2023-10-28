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
        self._calc_basis_mol = 1  # Basis of calculation - 1 mol

    def create_system(self) -> tc:
        """
        Creates a system based on the class TernaryCalculate.
        Check the TernaryCalculate script for more info.
        """
        system = tc()
        return system

    def calc_ternary_data(self, system: tc, to_df: bool = False, **kwarg) -> np.ndarray:
        """
        Calculates the ternary data for the given system.

        Paramters
        =========
        system: TernaryCalculate
            The system with solute, solvent, antisolvent.

        to_df: bool, optional
            If set to True, returns the ternary data in DataFrame format.
            If set to False, returns in array format.
            Default is set to False.

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

        # self.ternary_data = ternary_data
        # SHOULD I INITIALIZE THE SYSTEM?

        if to_df:
            ternary_data = pandas.DataFrame(
                ternary_data,
                index=[
                    "solute mole frac",
                    "solvent mole frac",
                    "antisolvent mole frac",
                ],
            )

        return ternary_data

    def init_mol_frac(self, to_df: bool = False) -> np.ndarray:
        """
        Returns the inital composition of the system.
        Initially, the system has only the solute and solvent.

        Paramters
        =========
        to_df: bool, optional
            If set to True, returns the ternary data in DataFrame format.
            If set to False, returns in array format.
            Default is set to False.

        Returns
        =======
        ndarray
            Array with inital mole fractions.
            Has shape of (1, 3)
        """
        init_mol = self.ternary_data[0, :]

        if to_df:
            init_mol = pandas.DataFrame(
                init_mol, index=["solute init", "solvent init", "antisolvent init"]
            )

        return init_mol
