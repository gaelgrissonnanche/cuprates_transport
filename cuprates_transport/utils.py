from scipy.constants import electron_mass, physical_constants, Boltzmann, elementary_charge
from numpy import pi

# Units ////////
meV = physical_constants["electron volt"][0] * 1e-3     # 1 meV in Joule
m0 = electron_mass                                      # in kg
Angstrom = 1e-10                                        # 1 A in meters
picosecond = 1e-12                                      # 1 ps in seconds

## Constant //////
e = elementary_charge # C
kB = Boltzmann # J / K
kB = kB / meV # meV / K

## This coefficient takes into accound all units and constant to prefactor Chambers formula
units_chambers = 2 * e**2 / (2*pi)**3 * picosecond / Angstrom**2
