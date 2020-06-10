from numpy import pi
from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## ONE BAND Matt et al. ///////////////////////////////////////////////////////
params = {
    "bandname": "LargePocket",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "t": 190,
    "tp": -0.136,
    "tpp": 0.068,
    "tz": 0.07,
    "tz2": 0.07,
    "tz3": 0.07,
    "tz4": 0.07,
    "mu": -0.83,
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
    "gamma_k": 0,#72.2,
    "gamma_dos_max": 0,
    "gamma_step": 70.0,
    "phi_step": pi/6.0,
    "power": 11.2,
    "factor_arcs": 1,
}

## Create Bandstructure object
bandObject = BandStructure(**params)

## Discretize Fermi surface
# bandObject.setMuToDoping(0.4)
# print(bandObject.mu)
bandObject.runBandStructure(printDoping=True)
# bandObject.mc_func()
# print("mc = " + "{:.3f}".format(bandObject.mc))
# bandObject.figMultipleFS2D()
# bandObject.figDiscretizeFS2D()

# import cProfile
# cProfile.run("for i in range(100): bandObject.discretize_FS()", sort="cumulative")

## Compute conductivity
condObject = Conductivity(bandObject, **params)
# condObject.runTransport()
# condObject.omegac_tau_func()
# print("omega_c * tau = " + "{:.3f}".format(condObject.omegac_tau))
# condObject.figScatteringPhi(kz=0)
# condObject.figScatteringPhi(kz=pi/bandObject.c)
# condObject.figScatteringPhi(kz=2*pi/bandObject.c)
# condObject.figArcs()

## Compute ADMR
amro1band = ADMR([condObject], **params)
amro1band.runADMR()
amro1band.fileADMR(folder="sim/NdLSCO_0p24")
amro1band.figADMR(folder="sim/NdLSCO_0p24")


