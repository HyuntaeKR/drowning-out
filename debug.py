import numpy as np
import pandas
import matplotlib.pyplot as plt
import ternary
import json

from cosmosac2 import COSMOMolecule, COSMOSAC
from ternary_calculate import TernaryCalculate as tc
from antisolvent_calculate import AntisolventCalculate as ac

# Solute properties
solute = {
    "temp_melt": 370.9,
    "enth_fus": 20700,
    "file": "./cosmo_file/UD1078.cosmo",
    "name": "GLUTARIC_ACID",
}
# Solvent properties
solvent = {
    "temp_melt": 184.552,
    "enth_fus": 9372.16,
    "file": "./cosmo_file/UD34.cosmo",
    "name": "1-BUTANOL",
}

# Antisolvent properties
antisolvent = {
    "temp_melt": 298.7,
    "enth_fus": 11720,
    "file": "./cosmo_file/UD69.cosmo",
    "name": "ACETIC_ACID",
}

system = tc()
system.add_solute(**solute)
system.add_solvent(**solvent)
system.add_antisolvent(**antisolvent)
antisolv_calculator = ac(system, trace=False)

system.plot_ternary(antisolv_calculator.ternary_data)