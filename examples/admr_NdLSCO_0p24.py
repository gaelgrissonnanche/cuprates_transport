from numpy import pi, deg2rad
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## ONE BAND Matt et al. ///////////////////////////////////////////////////////
params = {
    "band_name": "LargePocket",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 190,
    "band_params":{"mu":-0.83, "t": 1, "tp":-0.136, "tpp":0.068, "tz":0.07},
    "res_xy": 20,
    "res_z": 7,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 15, 30, 45],
    "gamma_0": 15.1,
    "gamma_k": 69,
    "gamma_dos_max": 0,
    "power": 12,
    "factor_arcs": 1,
    # "espilon_z": "- 2*tz*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)",
    # "a1": 8.49772756,
    # "a2": -141.605389,
    # "a3": 830.306324,
    # "a4": -1644.49799,
    # "a5": 598.753854,
}


## Create Bandstructure object
bandObject = BandStructure(**params)

## Discretize Fermi surface
# bandObject.setMuToDoping(0.4)
# print(bandObject["mu"])
bandObject.runBandStructure(printDoping=True)
# bandObject.mc_func()
# print("mc = " + "{:.3f}".format(bandObject.mc))
# bandObject.figMultipleFS2D()
# # bandObject.figDiscretizeFS2D()


# ## Compute conductivity
condObject = Conductivity(bandObject, **params)
condObject.runTransport()
# condObject.figScatteringColor(gamma_max=60)
# # condObject.omegac_tau_func()
# # print("omega_c * tau = " + "{:.3f}".format(condObject.omegac_tau))
# # condObject.figScatteringPhi(kz=0)
# # condObject.figScatteringPhi(kz=pi/bandObject.c)
# # condObject.figScatteringPhi(kz=2*pi/bandObject.c)
# # condObject.figArcs()

# ## Compute ADMR
amro1band = ADMR([condObject], **params)
amro1band.runADMR()
amro1band.fileADMR(folder="sim/NdLSCO_0p24")
amro1band.figADMR(folder="sim/NdLSCO_0p24")
