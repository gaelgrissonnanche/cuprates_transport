import numpy as np
import random
from copy import deepcopy
from genetic import genetic_search
import genetic_utils as utils
import os
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

init_member = {
    "bandname": "HolePocket",
    "a": 5.46,
    "b": 5.46,
    "c": 23.2,
    "t": 181.25,
    "tp": -0.4166,
    "tpp": -0.02193,
    "tz": 0.02,
    "tz2": 0.00,
    "mu": -1.345,
    "fixdoping": 0.24,
    "numberOfKz": 7,
    "mesh_ds": 1 / 20,
    "Ntime":500,
    "T": 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0], #, 20, 28, 36, 44],
    "gamma_0": 0.7751659419489383,
    "gamma_k": 10,
    "gamma_dos_max": 0,
    "power": 2,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 4.2,
    "data_p": 0.25,
}


ranges_dict = {
    # "t": [160,200],
    # "tp": [-0.45,-0.38],
    # "tpp": [-0.04,-0.01],
    "tz": [-0.20,0.20],
    # "mu": [-0.85,-0.3],
    "gamma_0": [1,7],
    "gamma_k": [1,30],
    "power":[0.1, 20],
    # "gamma_dos_max": [10.0,300.0],
    # "factor_arcs" : [1, 300],
}

# ## Data Nd-LSCO 0.24  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# data_dict = {}  # keys (T, phi), content [filename, theta, rzz, theta_cut]
# data_dict[25, 0] = ["../data_NdLSCO_0p25/0p25_0degr_45T_25K.dat", 0, 1, 90]
# data_dict[25, 15] = ["../data_NdLSCO_0p25/0p25_15degr_45T_25K.dat", 0, 1, 90]
# data_dict[25, 30] = ["../data_NdLSCO_0p25/0p25_30degr_45T_25K.dat", 0, 1, 90]
# data_dict[25, 45] = ["../data_NdLSCO_0p25/0p25_45degr_45T_25K.dat", 0, 1, 90]

# data_dict[20, 0] = ["../data_NdLSCO_0p25/0p25_0degr_45T_20K.dat", 0, 1, 90]
# data_dict[20, 15] = ["../data_NdLSCO_0p25/0p25_15degr_45T_20K.dat", 0, 1, 90]
# data_dict[20, 30] = ["../data_NdLSCO_0p25/0p25_30degr_45T_20K.dat", 0, 1, 90]
# data_dict[20, 45] = ["../data_NdLSCO_0p25/0p25_45degr_45T_20K.dat", 0, 1, 90]

# data_dict[12, 0] = ["../data_NdLSCO_0p25/0p25_0degr_45T_12K.dat", 0, 1, 83.5]
# data_dict[12, 15] = ["../data_NdLSCO_0p25/0p25_15degr_45T_12K.dat", 0, 1, 83.5]
# data_dict[12, 45] = ["../data_NdLSCO_0p25/0p25_45degr_45T_12K.dat", 0, 1, 83.5]

# data_dict[6, 0] = ["../data_NdLSCO_0p25/0p25_0degr_45T_6K.dat", 0, 1, 73.5]
# data_dict[6, 15] = ["../data_NdLSCO_0p25/0p25_15degr_45T_6K.dat", 0, 1, 73.5]
# data_dict[6, 45] = ["../data_NdLSCO_0p25/0p25_45degr_45T_6K.dat", 0, 1, 73.5]


## Data Nd-LSCO 0.21  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
data_dict = {}  # keys (T, phi), content [filename, theta, rzz, theta_cut]
data_dict[4.2, 0] = ["../data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_0.dat", 0, 1, 80]
data_dict[4.2, 20] = ["../data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_20.dat", 0, 1, 80]
data_dict[4.2, 28] = ["../data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_28.dat", 0, 1, 80]
data_dict[4.2, 36] = ["../data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_36.dat", 0, 1, 80]
data_dict[4.2, 44] = ["../data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_44.dat", 0, 1, 80]


# Play
# genetic_search(init_member,ranges_dict, data_dict, folder="../sim/Tl2201_Tc_20K",
#                 population_size=50, N_generation=100, mutation_s=0.1, crossing_p=0.9)


# utils.save_member_to_json(init_member, folder="../data_NdLSCO_0p25")
# init_member = utils.load_member_from_json(
#     "../sim/Tl2201_Tc_20K",
#     "data_p0.25_T4.2_fit_p0.240_T0_B45_t192.0_mu-1.360_tp-0.407_tpp-0.039_tz-0.003_tzz0.000_HolePocket_gzero0.6_gdos0.0_gk0.0_pwr5.5_arc1.0"
# )
utils.fig_compare(init_member, data_dict, folder="../sim/Tl2201_Tc_20K")



# class ObjectView():
#     def __init__(self,dictionary):
#         self.__dict__.update(dictionary)