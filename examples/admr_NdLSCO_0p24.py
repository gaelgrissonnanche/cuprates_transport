from numpy import pi, deg2rad, linalg
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
    "resolution": [21, 21, 7],
    "N_time": 1000,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 140,
    "Btheta_step": 5,
    "Bphi_array": [0],
    "scattering_params":{"constant": {"gamma_0":12.595},
                         "cos2phi": {"gamma_k": 63.823, "power": 12}}
}


## Create Bandstructure object
tband = time()
bandObject = BandStructure(**params)
bandObject.march_square = True


## Discretize Fermi surface
bandObject.runBandStructure(printDoping=False)

print("time structure = " + str(time()-tband) + " s")

## Compute conductivity
ttransport = time()
condObject = Conductivity(bandObject, **params)
condObject.runTransport()

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

# ## Compute ADMR
tadmr = time()
admr1band = ADMR([condObject], **params)
admr1band.runADMR()
print("time admr = " + str(time() - tadmr) + " s")

print("time total = " + str(time() - ttot) + " s")

# amro1band.fileADMR(folder="sim/NdLSCO_0p24")
admr1band.figADMR(fig_save=False) #(folder="sim/NdLSCO_0p24")