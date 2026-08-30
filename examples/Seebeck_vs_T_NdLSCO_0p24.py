from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
from cuprates_transport.admr import ADMR
import numpy as np
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## ADMR absolute AMPGO /////////////////////////////////////////////////////
params = {
    "band_name": "Nd-LSCO",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 160,
    "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
    "resolution": [21, 21, 7],
    "k_max": [np.pi, np.pi, 2*np.pi],
    "band_model": "-mu - 2*t*(cos(a*kx) + cos(b*ky))" + \
                  "- 4*tp*cos(a*kx)*cos(b*ky)" + \
                  "- 2*tpp*(cos(2*a*kx) + cos(2*b*ky))" + \
                  "- 2*tz*(cos(a*kx) - cos(b*ky))**2*cos(c*kz/2)*cos(a*kx/2)*cos(b*ky/2)",
    "Bamp": 0,
    "T" : 5,
    "dfdE_cut_percent": 0.001,
    "N_epsilon": 12,
    "scattering_models":["isotropic", "cos2phi"],
    "scattering_params":{"gamma_0":12.595, "gamma_k": 63.823, "power": 12},
}


## Create Bandstructure object
bandObject = BandStructure(**params)
bandObject.runBandStructure(printDoping=True)

## Compute conductivity
condObject = Conductivity(bandObject, **params)
condObject.runTransport()

## Resistivity >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
rho = np.linalg.inv(condObject.sigma).transpose()
print("Nd-LSCO -------------")
print("rhoxx =", rho[0,0]*1e8, "uOhm.cm")

## Seebeck vs Temperature
T_array = np.arange(5, 50, 5)
for T in T_array:
    condObject.T = T
    condObject.runTransport()
    sigma = condObject.sigma
    alpha = condObject.alpha
    Sxx = alpha[0,0] / sigma[0,0] # V / K

    print("T="+ str(condObject.T) + " : S / T =", Sxx*1e6/condObject.T, "uV / K^2")
