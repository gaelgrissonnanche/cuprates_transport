from numpy import pi, deg2rad
import numpy as np
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
from scipy.constants import Boltzmann, hbar, elementary_charge, physical_constants, electron_mass
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Results
# m* = 3.8099736731790803
# RH (Chambers) = -0.51233 mm^3/C
# RH (Drude) =  -1.036 mm^3/C
# rho_xx (Chambers) = 14.072087197973831 uOhm.cm
# rho_xx (Drude) = 28.26566466784027 uOhm.cm
# S/T (Chambers) =  -0.04835742884232023 uV / K^2 T_F = 5862.571847857093  K
# S/T (Drude) =  -0.04889228720420854 uV / K^2 T_F = 5798.43810091454  K


##!!!!!!! WARNING !!!!!!!!##
# You need to set kz_max = pi / c, instead of 2pi / c for it work. #####

params = {
    "band_name": "Free electrons",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 1000,
    "band_params":{"mu":-0.5, "t": 1},
    "band_model": "mu + t*((kx)**2+(ky)**2)", #+ t*((kx+pi/a)**2+(ky-pi/b)**2) + t*((kx-pi/a)**2+(ky-pi/b)**2)",
    "resolution": [300, 300, 3],
    "k_max": [pi, pi, pi],
    "dfdE_cut_percent": 0.001,
    "N_epsilon": 30,
    "N_time": 1000,
    "T" : 0,
    "Bamp": 0.1,
    "scattering_models":["isotropic"],
    "scattering_params":{"gamma_0":12.595},
}


## Units ////////
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule

a = params["a"]* 1e-10
c = params["c"]* 1e-10
## Volume between two planes in the unit cell
V = a**2 * c

## Create Bandstructure object
bandObject = BandStructure(**params)

## Discretize Fermi surface
bandObject.runBandStructure()
# bandObject.figMultipleFS2D()
# bandObject.figDiscretizeFS3D()


kf = np.sqrt(bandObject.kf[0,0]**2+bandObject.kf[1,0]**2)/1e-10 # m
# vf = np.sqrt(bandObject.vf[0,0]**2+bandObject.vf[1,0]**2)*1e-10/hbar

E_F = params["band_params"]["mu"]*params["energy_scale"]*meV
m_star =hbar**2*kf**2/(2*np.abs(E_F)) # comes from E_F = p_F**2 / (2 * m_star)
print("m* = " + str(m_star/electron_mass))
# bandObject.mass_func()
# print("m* = " + str(bandObject.mass))
# m_star = bandObject.mass * electron_mass

carrier_density = bandObject.n / V # in m^-3


## Compute rhoxx and RH >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
condObject = Conductivity(bandObject, **params)
condObject.runTransport()
sigma_xx = condObject.sigma[0,0]


if params["Bamp"] != 0:
    sigma_xy = condObject.sigma[1,0]
    rho_xy = - sigma_xy / (sigma_xx**2 +sigma_xy**2)
    RH = rho_xy / params["Bamp"] * 1e9
    print("RH (Chambers) = " + str(np.round(RH, 5)) + " mm^3/C")
    RH = 1/ (carrier_density * - elementary_charge) * 1e9
    print("RH (Drude) = ", str(np.round(RH, 5)) + " mm^3/C")

if params["Bamp"] != 0:
    rho_xx = sigma_xx / (sigma_xx**2 +sigma_xy**2) * 1e8
else:
    rho_xx = 1 / sigma_xx * 1e8
print("rho_xx (Chambers) = " + str(rho_xx) + " uOhm.cm")
rho_xx = m_star * condObject.gamma_0 * 1e12 / (carrier_density*elementary_charge**2)
print("rho_xx (Drude) = " + str(rho_xx*1e8) + " uOhm.cm")


## Compute Seebeck and T_F >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Seebeck from Chambers
condObject.Bamp = 0
condObject.T = 0.1
condObject.runTransport()
sigma = condObject.sigma
alpha = condObject.alpha
Sxx = alpha[0,0] / sigma[0,0] # V / K``
T_Fermi =  1 / (np.abs(Sxx)/condObject.T * 3 / np.pi**2 * elementary_charge / Boltzmann)
print("S/T (Chambers) = ", Sxx*1e6/condObject.T, "uV / K^2", "T_F =", T_Fermi, " K")

# Calculate the Fermi energy for 2D system
E_F = (hbar**2 * np.pi * carrier_density * c) / m_star
# Convert the Fermi energy to Fermi temperature
T_F = E_F / Boltzmann

S_formula =  1 / (T_F/condObject.T * 3 / np.pi**2 * elementary_charge / Boltzmann)
print("S/T (Drude) = ", -S_formula/condObject.T*1e6, "uV / K^2", "T_F =", T_F, " K")