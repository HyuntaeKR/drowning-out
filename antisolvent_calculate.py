import numpy as np
from numpy.typing import ArrayLike
import pandas
import matplotlib.pyplot as plt
import ternary

from cosmosac2 import COSMOMolecule, COSMOSAC
from ternary_calculate import TernaryCalculate as tc


def init_mole_frac(ternary_data: np.ndarray) -> np.ndarray:
    """
    Returns the inital composition of the system.
    Initially, the system has only the solute and solvent.

    Paramters
    ---------
    ternary_data: np.ndarray, shape=(ngrid, 3)
        Array with the ternary phase mole fraction.

    Returns
    -------
    ndarray: shape=(1, 3)
        Array with inital mole fractions.
    """
    init_fraction = ternary_data[0, :]

    # Code that makes the initial mole fraction into a DataFrame
    # if to_df:
    #     init_fraction = pandas.DataFrame(
    #         init_fraction, index=["solute init", "solvent init", "antisolvent init"]
    #     )

    return init_fraction


def calc_ratios(ternary_data: np.ndarray) -> dict:
    """
    Calculates the capacity ratio and antisolvent ratio based on the cosmo calculation.
    Capacity ratio is mole fraction of solute divided by that of solvent.
    Antisolvent ratio is mole fraction of antisolvent divided by that of solvent.

    Paramters
    ---------
    ternary_data: np.ndarray, shape=(1, 3)
        Array with the ternary phase mole fraction.

    Returns
    -------
    dict
        Keys of [capacity ratio, antisolvent ratio].
        Corresponding values are arrays of shape (ngrid, 1) with the ratio values.
    """
    capacity_ratio = np.zeros(np.shape(ternary_data)[0])  # (ngrid, 1)
    # First and last elements are omitted due to 'division by zero'
    capacity_ratio[1:-1] = ternary_data[1:-1, 0] / ternary_data[1:-1, 1]
    # Fill NaN values for first and last ratios
    capacity_ratio[0] = np.nan
    capacity_ratio[-1] = np.nan

    antisolv_ratio = np.zeros(np.shape(ternary_data)[0])
    antisolv_ratio[1:-1] = ternary_data[1:-1, 2] / ternary_data[1:-1, 1]
    antisolv_ratio[0] = np.nan
    antisolv_ratio[-1] = np.nan

    ratios = {"capacity_ratio": capacity_ratio, "antisolv_ratio": antisolv_ratio}

    return ratios


def calc_moles(init_frac: np.ndarray, ratios: dict) -> dict:
    """
    Calculate the mole values needed for antisolvent screening.
    Includes mole of antisolvent to add experimentally,
    corresponding capacity of solute dissolution,
    mole of precipitated solute.

    Parameters
    ----------
    init_frac: np.ndarray, shape=(1, 3)
        Initial mole fraction of solute-solvent.

    ratios: dict
        Keys of [capacity_ratio, antisolv_ratio].
        Corresponding values are arrays of shape (ngrid, 1)
        with the ratio values.

    Returns
    -------
    dict
        Keys of [add_antisov_mole, capacity_mole, precip_mole].
        Values are arrays of shape=(ngrid, 1)
    """
    # Set basis of calculation
    calc_basis_mole = 1

    # Unpack ratios
    capacity_ratio, antisolv_ratio = ratios

    # Create format for dict value arrays
    array_format = np.zeros(len(ratios["capacity ratio"]))
    array_format[0] = np.nan
    array_format[-1] = np.nan

    add_antisolv_mole = array_format
    add_antisolv_mole[1:-1] = calc_basis_mole * init_frac[1] * antisolv_ratio

    capacity_mole = array_format
    capacity_mole[1:-1] = calc_basis_mole * init_frac[1] * capacity_ratio

    precip_mole = array_format
    precip_mole[1:-1] = calc_basis_mole * init_frac[0] - capacity_mole

    moles = {
        "add_antisolv_mole": add_antisolv_mole,
        "capacity_mole": capacity_mole,
        "precip_mole": precip_mole,
    }

    return moles


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
        )  # ternary_data has shape of (ngrid, 3)

        self.ternary_data = ternary_data
