from numpy import pi, deg2rad
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#



## ADMR absolute AMPGO /////////////////////////////////////////////////////
params = {
    "band_name": "Nd-LSCO",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 190,
    "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
    "res_xy": 20,
    "res_z": 7,
    "N_time": 1000,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 110,
    "Btheta_step": 5,
    "Bphi_array": [0],
    "gamma_0": 15,
    "gamma_k": 75,
    "power": 12,
}

## Create Bandstructure object
bandObject = BandStructure(**params)

## Discretize Fermi surface
bandObject.runBandStructure(printDoping=True)

## Compute conductivity
condObject = Conductivity(bandObject, **params)
condObject.runTransport()

## Compute ADMR
amro1band = ADMR([condObject], **params)
amro1band.runADMR()
print("rzz(75)  = " + str(amro1band.rhozz_array[0][-8]))
print("rzz(105) = " + str(amro1band.rhozz_array[0][-2]))
print("delta    = " + str(amro1band.rhozz_array[0][-2] - amro1band.rhozz_array[0][-8]))
# amro1band.fileADMR(folder="sim/NdLSCO_0p24")
# amro1band.figADMR(folder="sim/NdLSCO_0p24")
