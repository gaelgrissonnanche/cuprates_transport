from numpy import pi, deg2rad, linalg
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# ## Fudge ! ADMR absolute AMPGO ///////////////////////////////////////////////////
# params = {
#     "band_name": "LargePocket",
#     "a": 3.75,
#     "b": 3.75,
#     "c": 13.2,
#     "energy_scale": 190,
#     "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
#     # "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06573192},
#     "fudge_vF": "1 + 7.24 * cos(2*atan2(ky, kx))**12",
#     "res_xy": 20,
#     "res_z": 7,
#     "T" : 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 120,
#     "Btheta_step": 2,
#     "Bphi_array": [0, 15, 30, 45],
#     "gamma_0": 14.956,
#     "gamma_k": -3.89, # 75.79
#     "gamma_dos_max": 0,
#     "power": 2,
#     "factor_arcs": 1,
#     # "e_z": "- 2*tz*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)",
# }

# ## ADMR absolute Published Nature ////////////////////////////////////////////////
# params = {
#     "band_name": "Nd-LSCO",
#     "a": 3.75,
#     "b": 3.75,
#     "c": 13.2,
#     "energy_scale": 160,
#     "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
#     "res_xy": 50,
#     "res_z": 15,
#     "N_time": 1000,
#     "T" : 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 140,
#     "Btheta_step": 5,
#     "Bphi_array": [0],
#     "gamma_0": 12.595,
#     "gamma_k": 63.823,
#     "gamma_dos_max": 0,
#     "power": 12,
#     "factor_arcs": 1,
#     # "e_z":"+2*tz*cos(c*kz)"
#     # "e_z":"-2 * cos(c*kz/2)*(" +\
#     #             "+0.50 * tz  *  cos(kx * a / 2) * cos(ky * b / 2)" +\
#     #             "-0.25 * tz2 * (cos(3 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(3 * ky * b / 2))" +\
#     #             "-0.50 * tz3 *  cos(3 * kx * a / 2) * cos(3 * ky * b / 2)" +\
#     #             "+0.25 * tz4 * (cos(5 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(5 * ky * b / 2))" +\
#     #             ")",
# }


# ## ADMR absolute AMPGO /////////////////////////////////////////////////////
# params = {
#     "band_name": "Nd-LSCO",
#     "a": 3.75,
#     "b": 3.75,
#     "c": 13.2,
#     "energy_scale": 190,
#     "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
#     "res_xy": 40,
#     "res_z": 11,
#     "N_time": 1000,
#     "T" : 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 140,
#     "Btheta_step": 5,
#     "Bphi_array": [0, 15, 30, 45],
#     "gamma_0": 15,
#     "gamma_k": 75,
#     "gamma_dos_max": 0,
#     "power": 12,
#     "factor_arcs": 1,
#     # "e_z":"+2*tz*cos(c*kz)"
#     # "e_z":"-2 * cos(c*kz/2)*(" +\
#     #             "+0.50 * tz  *  cos(kx * a / 2) * cos(ky * b / 2)" +\
#     #             "-0.25 * tz2 * (cos(3 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(3 * ky * b / 2))" +\
#     #             "-0.50 * tz3 *  cos(3 * kx * a / 2) * cos(3 * ky * b / 2)" +\
#     #             "+0.25 * tz4 * (cos(5 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(5 * ky * b / 2))" +\
#     #             ")",
# }

## ADMR absolute 160 meV marching cube ///////////////////////////////////////////
params = {
    "band_name": "Nd-LSCO",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 160,
    "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.0614644},
    "res_xy": 40,
    "res_z": 11,
    "N_time": 500,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 15, 30, 45],
    "gamma_0": 11.8232, # 12.628 (25), 11.8232 (20), 10.4308 (12), 9.4409 (6)
    "gamma_k": 75.4018, # 65.6884 (25), 75.4018 (20), 74.5853 (12), 77.74 (6)
    "power": 11.5492, # 11.6813 (25), 11.5492 (20), 13.6351 (12), 14.2606
    # "e_z":"+2*tz*cos(c*kz)"
    # "e_z":"-2 * cos(c*kz/2)*(" +\
    #             "+0.50 * tz  *  cos(kx * a / 2) * cos(ky * b / 2)" +\
    #             "-0.25 * tz2 * (cos(3 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(3 * ky * b / 2))" +\
    #             "-0.50 * tz3 *  cos(3 * kx * a / 2) * cos(3 * ky * b / 2)" +\
    #             "+0.25 * tz4 * (cos(5 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(5 * ky * b / 2))" +\
    #             ")",
}

# ## Play /////////////////////////////////////////////////////
# params = {
#     "band_name": "LargePocket",
#     "a": 3.75,
#     "b": 3.75,
#     "c": 13.2,
#     "energy_scale": 190,
#     "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
#     "res_xy": 20,
#     "res_z": 7,
#     "T" : 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 110,
#     "Btheta_step": 5,
#     "Bphi_array": [0, 45],
#     "gamma_0": 32,
#     "gamma_k": 0,
#     "gamma_dos_max": 0,
#     "power": 1.5,
#     "factor_arcs": 1,
#     "gamma_step":31,
#     "phi_step":deg2rad(5),
#     # "espilon_z": "- 2*tz*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)",
#     # "a1": 8.49772756,
#     # "a2": -141.605389,
#     # "a3": 830.306324,
#     # "a4": -1644.49799,
#     # "a5": 598.753854,
# }

# ## ONE BAND Matt et al. ///////////////////////////////////////////////////////
# params = {
#     "band_name": "LargePocket",
#     "a": 3.75,
#     "b": 3.75,
#     "c": 13.2,
#     "energy_scale": 190,
#     "band_params":{"mu":-0.83, "t": 1, "tp":-0.136, "tpp":0.068, "tz":0.07},
#     "res_xy": 20,
#     "res_z": 7,
#     "T" : 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 90,
#     "Btheta_step": 5,
#     "Bphi_array": [0, 15, 30, 45],
#     "gamma_0": 15.1,
#     "gamma_k": 69,
#     "gamma_dos_max": 0,
#     "power": 12,
#     "factor_arcs": 1,
#     # "espilon_z": "- 2*tz*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)",
#     # "a1": 8.49772756,
#     # "a2": -141.605389,
#     # "a3": 830.306324,
#     # "a4": -144.49799,
#     # "a5": 598.753854,
# }

## Fermi arcs
# params = {
#     "bandname": "LargePocket",
#     "a": 3.75,
#     "b": 3.75,
#     "c": 13.2,
#     "energy_scale": 190,
#     "band_params": {
#         "mu": -0.7686829828700767,
#         "t": 1,
#         "tp": -0.136,
#         "tpp": 0.068,
#         "tz": 0.07
#     },
#     "res_xy": 100,
#     "res_z": 7,
#     "fixdoping": 0.205,
#     "T": 0,
#     "Bamp": 45,
#     "Btheta_min": 0.0,
#     "Btheta_max": 110,
#     "Btheta_step": 3.0,
#     "Bphi_array": [
#         0.0,
#         15,
#         30,
#         45.0
#     ],
#     "gamma_0": 15,
#     "gamma_k": 0,
#     "gamma_dos_max": 0,
#     "power": 12,
#     "factor_arcs": 10,
#     "data_T": 25,
#     "data_p": 0.21
# }

## Create Bandstructure object
bandObject = BandStructure(**params)
# bandObject.march_square = True

## Discretize Fermi surface
# bandObject.setMuToDoping(0.15)
# print(bandObject["mu"])
bandObject.runBandStructure(printDoping=True)
# bandObject.figDiscretizeFS3D()
# bandObject.mc_func()
# print("mc = " + "{:.3f}".format(bandObject.mc))
# bandObject.figMultipleFS2D()
# # bandObject.figDiscretizeFS2D()
# print(bandObject.kf.shape)

# ## Compute conductivity
condObject = Conductivity(bandObject, **params)
condObject.runTransport()
# condObject.figScatteringColor()
# condObject.omegac_tau_func()
# print("omega_c * tau = " + "{:.3f}".format(condObject.omegac_tau))
# condObject.figOnekft()
# # condObject.figScatteringPhi(kz=0)
# # condObject.figScatteringPhi(kz=pi/bandObject.c)
# # condObject.figScatteringPhi(kz=2*pi/bandObject.c)

# rho = linalg.inv(condObject.sigma).transpose()
# rhoxx = rho[0,0]
# rhoxy = rho[0,1]
# rhozz = rho[2,2]
# print("1band-------------")
# print("rhoxx =", rhoxx*1e8, "uOhm.cm")
# print("rhozz =", rhozz*1e5, "mOhm.cm")
# print("RH =", rhoxy * 1e9 / params["Bamp"], "mm^3 / C")

# ## Compute ADMR
amro1band = ADMR([condObject], **params)
amro1band.runADMR()
# amro1band.fileADMR(folder="sim/NdLSCO_0p24")
amro1band.figADMR(folder="sim/NdLSCO_0p24")
