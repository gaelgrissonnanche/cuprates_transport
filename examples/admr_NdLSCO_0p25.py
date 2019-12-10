from numpy import pi
from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#



## ONE BAND Horio et al. ///////////////////////////////////////////////////////
params = {
    "bandname": "LargePocket",
    "a": 3.74767,
    "b": 3.74767,
    "c": 13.2,
    "t": 190,
    "tp": -0.14,
    "tpp": 0.07,
    "tz": 0.07,
    "tz2": 0.00,
    "mu": -0.826,
    "fixdoping": 0.1,
    "numberOfKz": 7,
    "mesh_ds": 1/20,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 15, 30, 45],
    "gamma_0": 15.1,
    "gamma_k": 66,
    "gamma_dos_max": 0,
    "power": 12,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 25,
    "data_p": 0.24,
}

bandObject = BandStructure(**params)

## Discretize
# bandObject.setMuToDoping(0.4)
# print(bandObject.mu)
bandObject.doping(printDoping=True)
bandObject.discretize_FS()
bandObject.dos_k_func()

# bandObject.figDiscretizeFS2D()
# bandObject.figMultipleFS2D()

## Conductivity
condObject = Conductivity(bandObject, **params)
# condObject.figdfdE()
condObject.solveMovementFunc()
condObject.omegac_tau_func()
print("omega_c * tau = " + "{:.3f}".format(condObject.omegac_tau))
condObject.mc_func()
print("mc = " + "{:.3f}".format(condObject.mc))
# condObject.figScatteringPhi(kz=0)
# condObject.figScatteringPhi(kz=pi/bandObject.c)
# condObject.figScatteringPhi(kz=2*pi/bandObject.c)
# condObject.figArcs()


## ADMR
amro1band = ADMR([condObject], **params)
amro1band.runADMR()
amro1band.fileADMR(folder="sim/NdLSCO_0p25")
amro1band.figADMR(folder="sim/NdLSCO_0p25")
