import numpy as np
import pandas
import matplotlib.pyplot as plt
import ternary

from cosmosac2 import COSMOMolecule, COSMOSAC
from ternary_calculate import TernaryCalculate as tc


def _init_mole_frac(ternary_data: np.ndarray) -> np.ndarray:
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
    init_frac = ternary_data[0, :]

    # Code that makes the initial mole fraction into a DataFrame
    # if to_df:
    #     init_frac = pandas.DataFrame(
    #         init_frac, index=["solute init", "solvent init", "antisolvent init"]
    #     )

    return init_frac


def _calc_ratios(ternary_data: np.ndarray) -> dict:
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
        Corresponding values are arrays of shape (ngrid,) with the ratio values.
    """
    capacity_ratio = np.zeros(np.shape(ternary_data)[0])  # (ngrid,)
    # First and last elements are omitted due to 'division by zero'
    capacity_ratio[1:-1] = ternary_data[1:-1, 0] / ternary_data[1:-1, 1]
    # Fill NaN values for first and last ratios
    capacity_ratio[0] = np.nan
    capacity_ratio[-1] = np.nan

    antisolv_ratio = np.zeros(np.shape(ternary_data)[0])  # (ngrid,)
    antisolv_ratio[1:-1] = ternary_data[1:-1, 2] / ternary_data[1:-1, 1]
    antisolv_ratio[0] = np.nan
    antisolv_ratio[-1] = np.nan

    ratios = {"capacity_ratio": capacity_ratio, "antisolv_ratio": antisolv_ratio}

    return ratios


def _calc_moles(
    init_frac: np.ndarray, ratios: dict, calc_basis_mole: float = 1
) -> dict:
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
        Corresponding values are arrays of shape (ngrid,)
        with the ratio values.

    calc_basis_mole: float, optional
        Basis of calculation in unit if mole.
        Default value is set to 1 mole.

    Returns
    -------
    dict
        Keys of [add_antisov_mole, capacity_mole, precip_mole].
        Values are arrays of shape=(ngrid,)
    """

    # Unpack ratios
    capacity_ratio = ratios["capacity_ratio"]
    antisolv_ratio = ratios["antisolv_ratio"]

    add_antisolv_mole = np.zeros(len(ratios["capacity_ratio"]))  # (ngrid,)
    add_antisolv_mole[1:-1] = calc_basis_mole * init_frac[1] * antisolv_ratio[1:-1]
    add_antisolv_mole[0] = np.nan
    add_antisolv_mole[-1] = np.nan

    capacity_mole = np.zeros(len(ratios["capacity_ratio"]))  # (ngrid,)
    capacity_mole[1:-1] = calc_basis_mole * init_frac[1] * capacity_ratio[1:-1]
    capacity_mole[0] = np.nan
    capacity_mole[-1] = np.nan

    precip_mole = np.zeros(len(ratios["capacity_ratio"]))  # (ngrid,)
    precip_mole[1:-1] = calc_basis_mole * init_frac[0] - capacity_mole[1:-1]
    precip_mole[0] = np.nan
    precip_mole[-1] = np.nan

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
        self.system = system
        self._calc_basis_mole = 1  # Basis of calculation: 1 mol

        print(
            f"System: {system.mole_name[0]}-{system.mole_name[1]}-{system.mole_name[2]}"
        )
        print("Initializing system...")

        ternary_data = system.calculate(
            ngrid=kwarg.get("ngrid", 101), trace=kwarg.get("trace", False)
        )  # ternary_data has shape of (ngrid, 3)

        self.ternary_data = ternary_data
        # Get inital mole fraction
        self.init_frac = _init_mole_frac(self.ternary_data)

        print("Initialize complete!")

    def get_data(self, export: str = None, file_name: str = None) -> pandas.DataFrame:
        """
        Get the data needed for antisolvent screening as a DataFrame.

        Parameters
        ----------
        export: str, optional
            {"csv", "excel"}.
            Choose the format to export the data.
            Default is set to None.

        file_name: str, optional
            The file name for exporting data.

        Returns
        -------
        DataFrame
            DataFrame with the antisolvent addition data.
        """
        ratios = _calc_ratios(self.ternary_data)
        moles = _calc_moles(
            self.init_frac, ratios, calc_basis_mole=self._calc_basis_mole
        )

        ratios = pandas.DataFrame(ratios)
        moles = pandas.DataFrame(moles)

        data = pandas.concat([ratios, moles], axis=1)

        if export == None:
            return data
        elif export == "csv":
            # When exporting as a data file,
            # the ternary data is added.
            ternary_data_export = pandas.DataFrame(
                self.ternary_data,
                columns=[
                    "solute_mol_fraction",
                    "solvent_mol_fraction",
                    "antisolvent_mol_fraction",
                ],
            )
            data = pandas.concat([data, ternary_data_export], axis=1)
            data.to_csv(file_name)
            return data
        elif export == "excel":
            ternary_data_export = pandas.DataFrame(
                self.ternary_data,
                columns=[
                    "solute_mol_fraction",
                    "solvent_mol_fraction",
                    "antisolvent_mol_fraction",
                ],
            )
            data = pandas.concat([data, ternary_data_export], axis=1)
            data.to_excel(file_name)
            return data
        else:
            raise Exception("Wrong data format!")

    def plot_antisolv(self) -> plt.figure:
        """
        Plots the solubility difference with respect to the amount of
        antisolvent added.
        """
        data = self.get_data()
        add_antisolv_mole = data["add_antisolv_mole"]
        precip_mole = data["precip_mole"]
        fig = plt.figure()
        plt.plot(
            add_antisolv_mole,
            precip_mole,
            label=f"antisolvent: {self.system.mole_name[2]}",
        )
        plt.hlines(0, 0, 100, colors="black", linestyles="dashed")
        plt.title(
            f"solute: {self.system.mole_name[0]}, solvent: {self.system.mole_name[1]}"
        )
        plt.ylim(
            [
                -self._calc_basis_mole * self.init_frac[0],
                self._calc_basis_mole * self.init_frac[0],
            ]
        )
        plt.xlabel("Antisolvent added [mol]")
        plt.ylabel("Solubility difference")
        plt.legend()

        return fig
