from numpy import pi
from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

params = {
    "bandname": "HolePocket",
    "a": 3.87,
    "b": 3.87,
    "c": 23.20,
    "t": 181,
    "tp": -0.28,
    "tpp": 0.14,
    "tppp" : 0,
    "tpppp" : 0,
    "tz": 0.05,
    # "tz": 0.0285*8,
    # "tz2": -0.0070,
    # "tz3": -0.0100,
    # "tz4": 0.005,
    "mu": -1.222,
    "fixdoping": 0.25,
    "numberOfKz": 7,
    "mesh_ds": 1 / 20,
    "Ntime": 500,
    "T": 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 20, 28, 36, 44],
    "gamma_0": 4.2,
    "gamma_k": 0,
    "power": 2,
    "gamma_dos_max": 0,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 4.2,
    "data_p": 0.25,
}

## ONE BAND Horio et al. /////////////////////////////////////////////////////////
bandObject = BandStructure(**params)

## Discretize
# bandObject.setMuToDoping(0.22)
bandObject.doping(printDoping=True)
bandObject.discretize_FS()
bandObject.dos_k_func()

bandObject.mc_func()
print("mc = " + "{:.3f}".format(bandObject.mc))
# bandObject.figDiscretizeFS2D()
bandObject.figMultipleFS2D()

## Conductivity
condObject = Conductivity(bandObject, **params)
# condObject.figdfdE()
condObject.solveMovementFunc()
condObject.omegac_tau_func()
print("omega_c * tau = " + "{:.3f}".format(condObject.omegac_tau))
# condObject.figScatteringPhi(kz=0)
# condObject.solveMovementFunc()
# condObject.figCumulativevft()

## ADMR
amro1band = ADMR([condObject], **params)
amro1band.runADMR()
amro1band.fileADMR(folder="sim/Tl2201_Tc_20K/")
amro1band.figADMR(folder="sim/Tl2201_Tc_20K/")
