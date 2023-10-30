import numpy as np
from numpy.typing import ArrayLike
import pandas
import matplotlib.pyplot as plt
import ternary

from cosmosac2 import COSMOMolecule, COSMOSAC
from ternary_calculate import TernaryCalculate as tc


class AntisolventCalculate:
    def __init__(self, system: tc, **kwarg) -> None:
        """
        Class calculating the properties of a ternary drowning-out crystallization system.
        Moreover, validates the antisolvent's effectivenss by calculating the
        solubility difference with respect to the amount of antisolvent added.

        Parameters
        ----------
        system: TernaryCalculate
            The ternary system with solute, solvent, antisolvent physical properties and
            cosmo files.

        Note
        ----
        See TernaryCalculate.calculate for kwarg info.
        """
        self._calc_basis_mol = 1  # Basis of calculation - 1 mol

        ternary_data = system.calculate(
            ngrid=kwarg.get("ngrid", 101), trace=kwarg.get("trace", True)
        )

        self.ternary_data = ternary_data

    def init_mole_frac(self, to_df: bool = False) -> np.ndarray:
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
