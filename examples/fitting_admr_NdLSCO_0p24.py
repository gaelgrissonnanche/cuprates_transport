from numpy import deg2rad
from cuprates_transport.fitting_admr import FittingADMR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## AMPGO ///////////////////////////////////////////////////////
init_member = {
    "bandname": "LargePocket",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 190,
    "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
    "res_xy": 20,
    "res_z": 7,
    # "fudge_vF": "1 + 5 * cos(2*atan2(ky, kx))**12",
    "fixdoping": 2,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [15, 30, 45],
    "gamma_0": 15,
    "gamma_k": 0,
    "gamma_dos_max": 0,
    "power": 12,
    "factor_arcs": 1,
    "data_T": 20,
    "data_p": 0.24,
}

## For step scattering rate
# init_member = {
#     "bandname": "LargePocket",
#     "a": 3.74767,
#     "b": 3.74767,
#     "c": 13.2,
#     "t": 190,
#     "tp": -0.136,
#     "tpp": 0.068,
#     "tz": 0.07,
#     "tz2": 0,
#     "tz3": 0,
#     "tz4": 0,
#     "mu": -0.83,
#     "fixdoping": 2,
#     "numberOfKz": 7,
#     "mesh_ds": 1 / 100,
#     "Ntime": 500,
#     "T" : 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 90,
#     "Btheta_step": 5,
#     "Bphi_array": [0, 45],
#     "gamma_0": 15.1,
#     "gamma_step": 15,
#     "phi_step": deg2rad(33),
#     "data_T": 25,
#     "data_p": 0.24,
# }


## For FIT
ranges_dict = {
    # "energy_scale": [170,210],
    # "tp": [-0.18,-0.1],
    # "tpp": [0.04,0.1],
    # "tppp": [-0.1,0.1],
    # "tpppp": [-0.05,0.05],
    # "tz": [0.04,0.1],
    # "tz2": [-0.07,0.07],
    # "tz3": [-0.07,0.07],
    # "tz4": [0,0.2],
    # "mu": [-1.0,-0.6],
    "gamma_0": [10,20],
    "gamma_k": [0,100],
    "power":[1, 20],
    # "gamma_dos_max": [0.1, 1000],
    # "factor_arcs" : [1, 300],
}


## Data Nd-LSCO 0.24 / H = 45T >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, rhozz_0] # rhozz_0 in SI units
data_dict[25, 0] = ["data/NdLSCO_0p24/0p25_0degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
data_dict[25, 15] = ["data/NdLSCO_0p24/0p25_15degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
data_dict[25, 30] = ["data/NdLSCO_0p24/0p25_30degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
data_dict[25, 45] = ["data/NdLSCO_0p24/0p25_45degr_45T_25K.dat", 0, 1, 90, 6.71e-5]

data_dict[20, 0] = ["data/NdLSCO_0p24/0p25_0degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
data_dict[20, 15] = ["data/NdLSCO_0p24/0p25_15degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
data_dict[20, 30] = ["data/NdLSCO_0p24/0p25_30degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
data_dict[20, 45] = ["data/NdLSCO_0p24/0p25_45degr_45T_20K.dat", 0, 1, 90, 6.55e-5]

data_dict[12, 0] = ["data/NdLSCO_0p24/0p25_0degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
data_dict[12, 15] = ["data/NdLSCO_0p24/0p25_15degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
data_dict[12, 45] = ["data/NdLSCO_0p24/0p25_45degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]

data_dict[6, 0] = ["data/NdLSCO_0p24/0p25_0degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
data_dict[6, 15] = ["data/NdLSCO_0p24/0p25_15degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
data_dict[6, 45] = ["data/NdLSCO_0p24/0p25_45degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]


# ## Data Nd-LSCO 0.24 / H = 35T >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, rhozz_0] # rhozz_0 in SI units
# data_dict[25, 15] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi15/NdLSCO_0p25_rho_c-vs-theta_25K_H35T_phi=15deg.txt", 0, 1, 90, 6.71e-5/1.005]
# data_dict[25, 30] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi30/NdLSCO_0p25_rho_c-vs-theta_25K_H35T_phi=30deg.txt", 0, 1, 90, 6.71e-5/1.005]
# data_dict[25, 45] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi45/NdLSCO_0p25_rho_c-vs-theta_25K_H35T_phi=45deg.txt", 0, 1, 90, 6.71e-5/1.005]

# data_dict[20, 15] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi15/NdLSCO_0p25_rho_c-vs-theta_20K_H35T_phi=15deg.txt", 0, 1, 90, 6.55e-5/1.005]
# data_dict[20, 30] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi30/NdLSCO_0p25_rho_c-vs-theta_20K_H35T_phi=30deg.txt", 0, 1, 90, 6.55e-5/1.005]
# data_dict[20, 45] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi45/NdLSCO_0p25_rho_c-vs-theta_20K_H35T_phi=45deg.txt", 0, 1, 90, 6.55e-5/1.005]

# data_dict[12, 15] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi15/NdLSCO_0p25_rho_c-vs-theta_12K_H35T_phi=15deg.txt", 0, 1, 77, 6.26e-5/1.005]
# data_dict[12, 45] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi45/NdLSCO_0p25_rho_c-vs-theta_12K_H35T_phi=45deg.txt", 0, 1, 77, 6.26e-5/1.005]

# data_dict[6, 15] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi15/NdLSCO_0p25_rho_c-vs-theta_6K_H35T_phi=15deg.txt", 0, 1, 66, 6.03e-5/1.005]
# data_dict[6, 45] = ["data/NdLSCO_0p24/all_data_NdLSCO0p24/phi45/NdLSCO_0p25_rho_c-vs-theta_6K_H35T_phi=45deg.txt", 0, 1, 66, 6.03e-5/1.005]

# ## Data Nd-LSCO 0.21  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, factor_to_SI]
# data_dict[25, 0] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_0.dat", 0, 2, 90, 35.8e-5]
# data_dict[25, 15] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_15.dat", 0, 2, 90, 35.8e-5]
# data_dict[25, 30] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_30.dat", 0, 2, 90, 35.8e-5]
# data_dict[25, 45] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_45.dat", 0, 2, 90, 35.8e-5]



fitObject = FittingADMR(init_member, ranges_dict, data_dict, folder="sim/NdLSCO_0p24",
                        method="ampgo", normalized_data=False)
fitObject.runFit()
# fitObject.fig_compare()
