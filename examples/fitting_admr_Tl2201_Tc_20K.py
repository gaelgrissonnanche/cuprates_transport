import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from cuprates_transport.fitting_admr import genetic_search, fit_search
import cuprates_transport.fitting_admr_utils as utils
import os
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## Peets
# init_member = {
#     "bandname": "HolePocket",
#     "a": 3.87,
#     "b": 3.87,
#     "c": 23.20,
#     "t": 181.25,
#     "tp": -0.4166,
#     "tpp": -0.0293,
#     "tppp" : 0.111,
#     "tpppp" : -0.0047,
#     "tz": -0.015,
#     "tz2": 0.00,
#     "mu": -1.222,
#     "fixdoping": 0.25,
#     "numberOfKz": 7,
#     "mesh_ds": 1 / 20,
#     "Ntime": 500,
#     "T": 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 90,
#     "Btheta_step": 5,
#     "Bphi_array": [0, 20, 28, 36, 44],
#     "gamma_0": 3,
#     "gamma_k": 0,
#     "power": 2,
#     "gamma_dos_max": 0,
#     "factor_arcs": 1,
#     "seed": 72,
#     "data_T": 4.2,
#     "data_p": 0.25,
# }


init_member = {
    "bandname": "HolePocket",
    "a": 3.87,
    "b": 3.87,
    "c": 23.20,
    "t": 181,
    "tp": -0.28,
    "tpp": 0.14,
    "tppp" : 0,
    "tpppp" : 0,
    "tz": 0.015,
    # "tz": 0.0285*8,
    # "tz2": -0.0070,
    # "tz3": -0.0100,
    # "tz4": 0.005,
    "mu": -1.222,
    "fixdoping": 0.25,
    "numberOfKz": 7,
    "mesh_ds": 1 / 20,
    "N_time": 500,
    "T": 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 44], #[0, 20, 28, 36, 44],
    "gamma_0": 3,
    "gamma_k": 3,
    "power": 6,
    "gamma_dos_max": 0,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 4.2,
    "data_p": 0.25,
}

## For GENETIC
# ranges_dict = {
#     # "t": [130,300],
#     "tp": [-0.7,0],
#     "tpp": [0,0.3],
#     "tppp": [-0.2,0.2],
#     "tpppp": [-0.2,0.2],
#     "fixdoping": [0.20, 0.35],
#     # "tz": [-0.1,0.1],
#     # "mu": [-1.8,-1.0],
#     "gamma_0": [1,10],
#     # "gamma_k": [0,30],
#     # "power":[1, 100],
#     # "gamma_dos_max": [0, 50],
#     # "factor_arcs" : [1, 300],
# }


## For FIT
ranges_dict = {
    "t": [130,300],
    # "tp": [-0.5,-0.2],
    # "tpp": [-0.14,0.14],
    # "tppp": [-0.1,0.1],
    # "tpppp": [-0.05,0.05],
    # "tz": [0,0.2],
    "tz2": [-0.2,0],
    "tz3": [-0.2,0],
    "tz4": [0, 0.2],
    # "mu": [-1.8,-1.0],
    # "gamma_0": [2,10],
    # "gamma_k": [0,30],
    # "power":[1, 100],
    # "gamma_dos_max": [0.1, 200],
    # "factor_arcs" : [1, 300],
}

## Data Tl2201 Tc = 20K (Hussey et al. 2003)  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, factor_to_SI]
data_dict[4.2, 0] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_0.dat", 0, 1, 70, 1e-2]
data_dict[4.2, 20] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_20.dat", 0, 1, 70, 1e-2]
data_dict[4.2, 28] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_28.dat", 0, 1, 70, 1e-2]
data_dict[4.2, 36] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_36.dat", 0, 1, 70, 1e-2]
data_dict[4.2, 44] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_44.dat", 0, 1, 70, 1e-2]


# Play
# genetic_search(init_member,ranges_dict, data_dict, folder="sim/Tl2201_Tc_20K",
#                 population_size=100, N_generation=1000, mutation_s=0.3, crossing_p=0.7, normalized_data=False)

# Play
# fit_search(init_member, ranges_dict, data_dict, folder="sim/Tl2201_Tc_20K", normalized_data=False)


# utils.save_member_to_json(init_member, folder="../data_NdLSCO_0p25")
# init_member = utils.load_member_from_json(
#     "sim/Tl2201_Tc_20K",
#     "data_p0.25_T4.2_fit_p0.200_T0_B45_t181.0_mu-1.375_tp-0.535_tpp0.300_tz0.015_tzz0.000_HolePocket_gzero6.5_gdos0.0_gk0.0_pwr2.0_arc1.0"
# )
utils.fig_compare(init_member, data_dict, folder="sim/Tl2201_Tc_20K", normalized_data=True)
