from cuprates_transport.fitting_admr import genetic_search, fit_search
import cuprates_transport.fitting_admr_utils as utils
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


init_member = {
    "bandname": "LargePocket",
    "a": 3.74767,
    "b": 3.74767,
    "c": 13.2,
    "t": 190,
    "tp": -0.14,
    "tpp": 0.07,
    "tz": 0.07,
    "tz2": 0,
    "tz3": 0,
    "tz4": 0,
    "mu": -0.826,
    "fixdoping": 0.24,
    "numberOfKz": 7,
    "mesh_ds": 1 / 20,
    "Ntime": 500,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 45],
    "gamma_0": 15.1,
    "gamma_k": 71.5,
    "gamma_dos_max": 0,
    "power": 11.5,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 25,
    "data_p": 0.24,
}


## For FIT
ranges_dict = {
    "t": [180,220],
    # "tp": [-0.2,-0.1],
    # "tpp": [-0.10,0.04],
    # "tppp": [-0.1,0.1],
    # "tpppp": [-0.05,0.05],
    # "tz": [0,0.2],
    "tz2": [-0.07,0.07],
    "tz3": [-0.07,0.07],
    # "tz4": [0,0.2],
    # "mu": [-1.8,-1.0],
    # "gamma_0": [5,20],
    "gamma_k": [0,150],
    "power":[1, 20],
    # "gamma_dos_max": [0.1, 200],
    # "factor_arcs" : [1, 300],
}


## Data Nd-LSCO 0.24  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, factor_to_SI]
data_dict[25, 0] = ["data/NdLSCO_0p25/0p25_0degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
data_dict[25, 15] = ["data/NdLSCO_0p25/0p25_15degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
data_dict[25, 30] = ["data/NdLSCO_0p25/0p25_30degr_45T_25K.dat", 0, 1, 90, 6.71e-5]
data_dict[25, 45] = ["data/NdLSCO_0p25/0p25_45degr_45T_25K.dat", 0, 1, 90, 6.71e-5]

data_dict[20, 0] = ["data/NdLSCO_0p25/0p25_0degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
data_dict[20, 15] = ["data/NdLSCO_0p25/0p25_15degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
data_dict[20, 30] = ["data/NdLSCO_0p25/0p25_30degr_45T_20K.dat", 0, 1, 90, 6.55e-5]
data_dict[20, 45] = ["data/NdLSCO_0p25/0p25_45degr_45T_20K.dat", 0, 1, 90, 6.55e-5]

data_dict[12, 0] = ["data/NdLSCO_0p25/0p25_0degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
data_dict[12, 15] = ["data/NdLSCO_0p25/0p25_15degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]
data_dict[12, 45] = ["data/NdLSCO_0p25/0p25_45degr_45T_12K.dat", 0, 1, 83.5, 6.26e-5]

data_dict[6, 0] = ["data/NdLSCO_0p25/0p25_0degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
data_dict[6, 15] = ["data/NdLSCO_0p25/0p25_15degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]
data_dict[6, 45] = ["data/NdLSCO_0p25/0p25_45degr_45T_6K.dat", 0, 1, 73.5, 6.03e-5]

# ## Data Nd-LSCO 0.21  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, factor_to_SI]
# data_dict[25, 0] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_0.dat", 0, 2, 90, 1]
# data_dict[25, 15] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_15.dat", 0, 2, 90, 1]
# data_dict[25, 30] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_30.dat", 0, 2, 90, 1]
# data_dict[25, 45] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_45.dat", 0, 2, 90, 1]


## Play
genetic_search(init_member,ranges_dict, data_dict, folder="sim/NdLSCO_0p25",
                population_size=200, N_generation=1000, mutation_s=0.3, crossing_p=0.7, normalized_data=True)

# Play
# fit_search(init_member, ranges_dict, data_dict, folder="sim/NdLSCO_0p25", normalized_data=False)

# utils.save_member_to_json(init_member, folder="data/NdLSCO_0p25")
# init_member = utils.load_member_from_json(
#     "sim/NdLSCO_0p25",
#     "data_p0.24_T25.0_fit_p0.240_T0_B45_t320.9_mu-0.977_tp-0.168_tpp0.120_tz0.020_tzz0.000_LargePocket_gzero5.3_gdos338.2_gk0.0_pwr12.0_arc1.0"
# )
# utils.fig_compare(init_member, data_dict, folder="sim/NdLSCO_0p25", normalized_data=True)

