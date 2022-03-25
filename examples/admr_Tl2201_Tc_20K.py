from numpy import pi
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# ## Peets et al.
# params = {
#     "bandname": "HolePocket",
#     "a": 3.87,
#     "b": 3.87,
#     "c": 23.20,
#     "energy_scale": 181.25,
#     "band_params":{"mu":-1.33, "t": 1, "tp":-0.42, "tpp":-0.021, "tppp": 0.111, "tpppp":-0.005, "tz":0.015},
#     "res_xy": 20,
#     "res_z": 5,
#     "epsilon_xy":  ( "-2 * t * (cos(a*kx) + cos(b*ky))"
#                    + "-4 * tp * cos(a*kx)*cos(b*ky)"
#                    + "-2 * tpp * (cos(2*a*kx) + cos(2*b*ky))"
#                    + "-2 * tppp * (cos(2 * kx * a) * cos(ky * b) + cos(kx * a) * cos(2 * ky * b))"
#                    + "-4 * tpppp * cos(2 * kx * a) * cos(2 * ky * b)"),
#     "fixdoping": 0.30,
#     "N_time": 500,
#     "T": 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 90,
#     "Btheta_step": 4,
#     "Bphi_array": [0, 20, 28, 36, 44],
#     "gamma_0": 3.7,
#     "gamma_k": 0,
#     "power": 6,
#     "data_T": 4.2,
#     "data_p": 0.30,
#     # "epsilon_z":"-2 * cos(c*kz/2)*(" +\
#     #             "+0.50 * tz  *  cos(kx * a / 2) * cos(ky * b / 2)" +\
#     #             "-0.25 * tz2 * (cos(3 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(3 * ky * b / 2))" +\
#     #             "-0.50 * tz3 *  cos(3 * kx * a / 2) * cos(3 * ky * b / 2)" +\
#     #             "+0.25 * tz4 * (cos(5 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(5 * ky * b / 2))" +\
#     #             ")",
# }

## Horio et al.
params = {
    "bandname": "HolePocket",
    "a": 3.87,
    "b": 3.87,
    "c": 23.20,
    "energy_scale": 181,
    "band_params":{"mu":-1.3280998884827857, "t": 1, "tp":-0.28, "tpp":0.14, "tz":0.015, "tz2":0.005002965958434592, "tz3":0.01499967470987841, "tz4":0.008479906699721694},
    "res_xy": 20,
    "res_z": 5,
    "fixdoping": 0.30,
    "N_time": 500,
    "T": 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 4,
    "Bphi_array": [0, 20, 28, 36, 44],
    "gamma_0": 3.77,
    "gamma_k": 0,
    "power": 6,
    "data_T": 4.2,
    "data_p": 0.30,
    "epsilon_z":"-2 * cos(c*kz/2)*(" +\
                "+0.50 * tz  *  cos(kx * a / 2) * cos(ky * b / 2)" +\
                "-0.25 * tz2 * (cos(3 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(3 * ky * b / 2))" +\
                "-0.50 * tz3 *  cos(3 * kx * a / 2) * cos(3 * ky * b / 2)" +\
                "+0.25 * tz4 * (cos(5 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(5 * ky * b / 2))" +\
                ")",
}

## ONE BAND Horio et al. /////////////////////////////////////////////////////////
bandObject = BandStructure(**params)

## Discretize
bandObject.setMuToDoping(0.30)
bandObject.runBandStructure(printDoping=True)

# bandObject.mc_func()
# print("mc = " + "{:.3f}".format(bandObject.mc))
# bandObject.figDiscretizeFS2D()
# bandObject.figMultipleFS2D()

## Conductivity
condObject = Conductivity(bandObject, **params)
# condObject.figdfdE()
# condObject.runTransport()
# condObject.omegac_tau_func()
# print("omega_c * tau = " + "{:.3f}".format(condObject.omegac_tau))
# condObject.figScatteringPhi(kz=0)
# condObject.solveMovementFunc()
# condObject.figCumulativevft()

## ADMR
amro1band = ADMR([condObject], **params)
amro1band.runADMR()
amro1band.fileADMR(folder="sim/Tl2201_Tc_20K/")
amro1band.figADMR(folder="sim/Tl2201_Tc_20K/")
