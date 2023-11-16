import numpy as np
import pandas
import matplotlib.pyplot as plt
import ternary

from cosmosac2 import COSMOMolecule, COSMOSAC

# Define constants
_R = 8.31446261815324  # Gas constant, [J/K/mol]


def _calc_sle_point(
    enth_fus: np.ndarray,
    temp_melt: np.ndarray,
    ratio: list,
    cosmo,
    temp: float = 298,
    trace: bool = True,
) -> np.ndarray:
    """
    Calculates a single point on the ternary phase diagram.
    The single point is where S and D is mixed in the given ratio from the parameter.
    The composition of A is calculated through iteration until converged within a desired
    error range.

    Parameters
    ----------
    enth_fus: np.ndarray
        Array of enthalpy of fusion.
        Has shape of (3,)

    temp_melt: np.ndarray
        Array of melting temperature.
        Has shape of (3,)

    ratio: list
        The ratio of S:D in the solvent mixture.
        Has shape of (2,)
        e.g., [0.2, 0.8]

    cosmo: type[COSMOSAC]
        The cosmo model with the system's components.

    temp: float
        The system temperature.
        Default value is room temperature of 298K.

    trace: bool
        If set to true, prints out the calculated composition.
        Default is set to True.

    Returns
    -------
    ndarray
        An array with compositions [x_A, x_S, x_D]
    """
    EPS = 1e-6  # Desired error
    # Guess initial composition by assuming ideality
    x_init = np.exp(-enth_fus / _R * (1 / temp - 1 / temp_melt))  # (3,)
    x_init[1:] = np.array([(1 - x_init[0]) * ratio[0], (1 - x_init[0]) * ratio[1]])
    x_old = x_init

    for iter in range(500):
        gamma_list = cosmo.gam(x_old, temp)
        # print(f"iter: {iter}, gamma value: {gamma_list}")
        x_new = np.exp(-enth_fus / _R * (1 / temp - 1 / temp_melt)) / gamma_list
        x_new[1:] = np.array([(1 - x_new[0]) * ratio[0], (1 - x_new[0]) * ratio[1]])
        # print(
        #     f"iter: {iter}, x_new: {np.round(x_new, 4)}, error: {abs(x_new[0] - x_old[0])}"
        # )
        # print()
        if abs(x_new[0] - x_old[0]) < EPS:
            break
        x_old = x_new

    # print("Iteration complete!")

    if trace:
        print(f"Composition >> ({x_new})")

    if any(x < 0 for x in x_new) or any(x > 1 for x in x_new):
        x_new = np.array([np.nan, np.nan, np.nan])

    return x_new


def _calc_sle_grid(ngrid: int, **kwarg) -> np.ndarray:
    """
    Calculated all the SLE points for the possible ratios.

    Parameters
    ----------
    ngrid: int
        The number of grids (i.e., number of ratios to calculate).

    Returns
    -------
    ndarray
        Shape of (ngrid, 3).
        Array with A, S, D compositions.

    Notes
    -----
    See 'calc_sle_point' to find keyword arguments.
    """
    ratiogrid = np.ones((ngrid, 2))
    ratiogrid[:, 0] = np.linspace(1, 0, ngrid)
    ratiogrid[:, 1] = 1 - ratiogrid[:, 0]

    result = np.empty((ngrid, 3))
    for idx, ratio in enumerate(ratiogrid):
        sle_point = _calc_sle_point(
            kwarg.get("enth_fus"),
            kwarg.get("temp_melt"),
            ratio,
            kwarg.get("cosmo"),
            kwarg.get("temp"),
            kwarg.get("trace"),
        )
        result[idx, :] = sle_point

    return result


class TernaryCalculate:
    def __init__(self, temp: float = 298):
        """
        Class calculating SLE composition for ternary systems.
        Based on COSMO-SAC model.

        Paramters
        =========
        temp: float
            System temperature with unit of Kelvin.
            Default value is room temperature 298K.
        """
        self.temp = temp  # [K]
        self.temp_melt = np.zeros(3)
        self.enth_fus = np.zeros(3)
        self.mole_file = ["solute_file", "solvent_file", "antisolvent_file"]
        self.mole_name = ["solute", "solvent", "antisolvent"]

    def add_solute(self, temp_melt: float, enth_fus: float, file: str, name: str):
        """
        Add solute molecule to the system.

        Parameters
        ----------
        temp_melt: float
            Melting temperature
        enth_fus: float
            Enthalpy of fusion

        Returns
        -------
        None

        Note
        ----
        See 'COSMOMolecule' to see keyword arguments.
        """
        self.temp_melt[0] = temp_melt
        self.enth_fus[0] = enth_fus
        self.mole_name[0] = name
        self.mole_file[0] = file

    def add_solvent(self, temp_melt: float, enth_fus: float, file: str, name: str):
        """
        Add solvent molecule to the system.

        Parameters
        ----------
        temp_melt: float
            Melting temperature
        enth_fus: float
            Enthalpy of fusion

        Returns
        -------
        None

        Note
        ----
        See 'COSMOMolecule' to see keyword arguments.
        """
        self.temp_melt[1] = temp_melt
        self.enth_fus[1] = enth_fus
        self.mole_name[1] = name
        self.mole_file[1] = file

    def add_antisolvent(self, temp_melt: float, enth_fus: float, file: str, name: str):
        """
        Add antisolvent molecule to the system.

        Parameters
        ----------
        temp_melt: float
            Melting temperature
        enth_fus: float
            Enthalpy of fusion

        Returns
        -------
        None

        Note
        ----
        See 'COSMOMolecule' to see keyword arguments.
        """
        self.temp_melt[2] = temp_melt
        self.enth_fus[2] = enth_fus
        self.mole_name[2] = name
        self.mole_file[2] = file

    def clear_solute(self):
        """
        Clears the solute component.
        """
        self.temp_melt[0] = 0
        self.enth_fus[0] = 0
        self.mole_name[0] = "solute"
        self.mole_file[0] = "solute_file"

    def clear_solvent(self):
        """
        Clears the solvent component.
        """
        self.temp_melt[1] = 0
        self.enth_fus[1] = 0
        self.mole_name[1] = "solvent"
        self.mole_file[1] = "solvent_file"

    def clear_antisolvent(self):
        """
        Clears the antisolvent component.
        """
        self.temp_melt[2] = 0
        self.enth_fus[2] = 0
        self.mole_name[2] = "antisolvent"
        self.mole_file[2] = "antisolvent_file"

    def _cosmo_model(self):
        """
        Generates and returns a cosmo model.
        """
        solute = COSMOMolecule(self.mole_file[0], name=self.mole_name[0])
        solvent = COSMOMolecule(self.mole_file[1], name=self.mole_name[1])
        antisolvent = COSMOMolecule(self.mole_file[2], name=self.mole_name[2])
        cosmo = COSMOSAC(solute, solvent, antisolvent, version=2019)

        return cosmo

    def calculate(self, ngrid: int = 21, trace: bool = True) -> np.ndarray:
        """
        Calculates data points for given ternary system.
        """
        ternary_data = _calc_sle_grid(
            ngrid=ngrid,
            enth_fus=self.enth_fus,
            temp_melt=self.temp_melt,
            cosmo=self._cosmo_model(),
            temp=self.temp,
            trace=trace,
        )

        return ternary_data

    def plot_ternary(self, ternary_data: np.ndarray) -> plt.figure:
        """
        Plots the ternary data.

        Returns
        -------
        figure
            Plot of the ternary diagram.
        """
        figure, tax = ternary.figure(scale=1.0)
        tax.boundary()
        tax.gridlines(multiple=0.2)
        tax.set_title(
            f"System of {self.mole_name[0]} - {self.mole_name[1]} - {self.mole_name[2]}"
        )

        tax.plot(
            ternary_data, linewidth=2.0, label=f"{self.mole_name[0]} crystallization"
        )
        tax.right_axis_label(f"{self.mole_name[1]} - {self.mole_name[0]}")
        tax.bottom_axis_label(f"{self.mole_name[2]} - {self.mole_name[0]}")
        tax.ticks()
        tax.legend(loc="upper right")

        return figure

    """
    Comments
    --------
    Add calculation validation.
    """
