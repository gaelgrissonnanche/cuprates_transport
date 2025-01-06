from numpy import pi, deg2rad, linalg
from scipy.constants import elementary_charge
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
from time import time
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

ttot = time()

## ADMR Published Nature 2021 ////////////////////////////////////////////////////
params = {
    "band_name": "Nd-LSCO",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 160,
    "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
    "resolution": [50, 50, 7],
    "k_max": [pi, pi, 2*pi],
    "N_time": 1000,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 140,
    "Btheta_step": 5,
    "Bphi_array": [0, 15, 30, 45],
    "scattering_models":["isotropic", "cos2phi"],
    "scattering_params":{"gamma_0":12.595, "gamma_k": 63.823, "power": 12},
}


## Create Bandstructure object
tband = time()
bandObject = BandStructure(**params)
bandObject.march_square = True

## Discretize Fermi surface
bandObject.runBandStructure(printDoping=False)
p = bandObject.p / (bandObject.a * bandObject.b * bandObject.c/2 * 1e-30)
n = bandObject.n / (bandObject.a * bandObject.b * bandObject.c/2 * 1e-30) 
print("n + p : ", n+p)
R_H = 1 / (1-p) / elementary_charge * 1e9
print("Calculated RH with p: ", R_H,  "mm^3 / C")
R_H = -1 / n / elementary_charge * 1e9
print("Calculated RH with n: ", R_H,  "mm^3 / C")

# bandObject.figDiscretizeFS2D()
print("time structure = " + str(time()-tband) + " s")

## Compute conductivity
ttransport = time()
condObject = Conductivity(bandObject, **params)
condObject.runTransport()
# condObject.figOnekft()
# condObject.figParameters()


## Compute resistivity
rho = linalg.inv(condObject.sigma).transpose()
rhoxx = rho[0,0]
rhoxy = rho[0,1]
rhozz = rho[2,2]
print("1band-------------")
print("rhoxx =", rhoxx*1e8, "uOhm.cm")
print("rhozz =", rhozz*1e5, "mOhm.cm")
print("RH =", rhoxy * 1e9 / params["Bamp"], "mm^3 / C")
print("time transport = " + str(time()-ttransport) + " s")

# # ## Compute ADMR
# tadmr = time()
# admr1band = ADMR([condObject], **params)
# admr1band.runADMR()
# print("time admr = " + str(time() - tadmr) + " s")

# print("time total = " + str(time() - ttot) + " s")

# # amro1band.fileADMR(folder="sim/NdLSCO_0p24")
# admr1band.figADMR(fig_save=False) #(folder="sim/NdLSCO_0p24")