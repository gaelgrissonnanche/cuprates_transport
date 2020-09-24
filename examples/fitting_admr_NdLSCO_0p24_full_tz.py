from cuprates_transport.fitting_admr import FittingADMR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ## ONE BAND Matt et al. ///////////////////////////////////////////////////////
# init_member = {
#     "bandname": "LargePocket",
#     "a": 3.75,
#     "b": 3.75,
#     "c": 13.2,
#     "energy_scale": 190,
#     "band_params":{"mu":-0.83, "t": 1, "tp":-0.136, "tpp":0.068, "tz":0.07, "tz2":0.07, "tz3":0.07, "tz4":0.07},
#     "res_xy": 20,
#     "res_z": 7,
#     "fixdoping": 2,
#     "T" : 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 90,
#     "Btheta_step": 5,
#     "Bphi_array": [0, 45],
#     "gamma_0": 15,
#     "gamma_k": 0,
#     "gamma_dos_max": 0,
#     "power": 12,
#     "factor_arcs": 1,
#     "data_T": 25,
#     "data_p": 0.21,
#     "epsilon_z":"-2 * cos(c*kz/2)*(" +\
#                 "+0.50 * tz  *  cos(kx * a / 2) * cos(ky * b / 2)" +\
#                 "-0.25 * tz2 * (cos(3 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(3 * ky * b / 2))" +\
#                 "-0.50 * tz3 *  cos(3 * kx * a / 2) * cos(3 * ky * b / 2)" +\
#                 "+0.25 * tz4 * (cos(5 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(5 * ky * b / 2))" +\
#                 ")",
# }


## ONE BAND Matt et al. ///////////////////////////////////////////////////////
init_member = {
    "bandname": "LargePocket",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 190,
    "band_params":{"mu":-0.3, "t": 1, "tp":-0.136, "tpp":0.136, "tz":0.25, "tz2":0, "tz3":0, "tz4":0},
    "res_xy": 20,
    "res_z": 7,
    "fixdoping": 2,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 45],
    "gamma_0": 100,
    "gamma_k": 0,
    "gamma_dos_max": 0,
    "power": 12,
    "factor_arcs": 1,
    "data_T": 25,
    "data_p": 0.21,
    "epsilon_z":"-2 * cos(c*kz/2)*(" +\
                "+0.50 * tz  *  cos(kx * a / 2) * cos(ky * b / 2)" +\
                "-0.25 * tz2 * (cos(3 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(3 * ky * b / 2))" +\
                "-0.50 * tz3 *  cos(3 * kx * a / 2) * cos(3 * ky * b / 2)" +\
                "+0.25 * tz4 * (cos(5 * kx * a / 2) * cos(ky * b / 2) + cos(kx * a / 2) * cos(5 * ky * b / 2))" +\
                ")",
}


## For FIT
ranges_dict = {
    # "t": [180,220],
    # "tp": [-0.2,-0.1],
    # "tpp": [-0.10,0.04],
    # "tppp": [-0.1,0.1],
    # "tpppp": [-0.05,0.05],
    # "tz": [0.06,0.08],
    "tz2": [0.05,0.08],
    "tz3": [0.05,0.08],
    "tz4": [0.05,0.08],
    # "mu": [-0.75,-0.9],
    "gamma_0": [10,25],
    "gamma_k": [0,150],
    "power":[1, 20],
    # "gamma_dos_max": [0.1, 200],
    # "factor_arcs" : [1, 300],
}


# ## Data Nd-LSCO 0.24  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, rhozz_0] # rhozz_0 in SI units
# data_dict[25, 0] = ["data/NdLSCO_0p24/0p25_0degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
# data_dict[25, 15] = ["data/NdLSCO_0p24/0p25_15degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
# data_dict[25, 30] = ["data/NdLSCO_0p24/0p25_30degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
# data_dict[25, 45] = ["data/NdLSCO_0p24/0p25_45degr_45T_25K.dat", 0, 1, 90, 6.71e-5]

# data_dict[20, 0] = ["data/NdLSCO_0p24/0p25_0degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
# data_dict[20, 15] = ["data/NdLSCO_0p24/0p25_15degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
# data_dict[20, 30] = ["data/NdLSCO_0p24/0p25_30degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
# data_dict[20, 45] = ["data/NdLSCO_0p24/0p25_45degr_45T_20K.dat", 0, 1, 90, 6.55e-5]

# data_dict[12, 0] = ["data/NdLSCO_0p24/0p25_0degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
# data_dict[12, 15] = ["data/NdLSCO_0p24/0p25_15degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
# data_dict[12, 45] = ["data/NdLSCO_0p24/0p25_45degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]

# data_dict[6, 0] = ["data/NdLSCO_0p24/0p25_0degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
# data_dict[6, 15] = ["data/NdLSCO_0p24/0p25_15degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
# data_dict[6, 45] = ["data/NdLSCO_0p24/0p25_45degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]

# ## Data Nd-LSCO 0.21  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, factor_to_SI]
data_dict[25, 0] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_0.dat", 0, 2, 90, 1]
data_dict[25, 15] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_15.dat", 0, 2, 90, 1]
data_dict[25, 30] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_30.dat", 0, 2, 90, 1]
data_dict[25, 45] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_45.dat", 0, 2, 90, 1]



fitObject = FittingADMR(init_member, ranges_dict, data_dict, folder="sim/NdLSCO_0p24",
                        method="differential_evolution")
# fitObject.runFit()
fitObject.bandObject.figMultipleFS2D()
fitObject.fig_compare()
