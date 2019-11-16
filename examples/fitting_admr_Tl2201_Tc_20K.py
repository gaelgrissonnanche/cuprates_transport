import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from cuprates_transport.fitting_admr import genetic_search, fit_search
import cuprates_transport.fitting_admr_utils as utils
import os
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# ## Yawen's
# init_member = {
#     "bandname": "HolePocket",
#     "a": 3.87,
#     "b": 3.87,
#     "c": 23.20,
#     "t": 180,
#     "tp": -0.47,
#     "tpp": -0.0252,
#     "tz": -0.005,
#     "tz2": 0,
#     "mu": -1.75,
#     "fixdoping": -2,
#     "numberOfKz": 7,
#     "mesh_ds": 1 / 20,
#     "Ntime": 500,
#     "T": 0,
#     "Bamp": 45,
#     "Btheta_min": 0,
#     "Btheta_max": 90,
#     "Btheta_step": 5,
#     "Bphi_array": [28], #[0, 20, 28, 36, 44],
#     "gamma_0": 1.8,
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
    "tppp" : -0.10,
    "tpppp" : -0.05,
    "tz": 0.015,
    "tz2": 0,
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
    "gamma_0": 4,
    "gamma_k": 0,
    "power": 2,
    "gamma_dos_max": 0,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 4.2,
    "data_p": 0.25,
}

## For GENETIC
# ranges_dict = {
#     # "t": [130,300],
#     "tp": [-0.5,0.5],
#     "tpp": [-0.2,0.2],
#     "tz": [-0.1,0.1],
#     # "mu": [-1.8,-1.0],
#     "gamma_0": [1,10],
#     # "gamma_k": [0,30],
#     # "power":[1, 100],
#     # "gamma_dos_max": [0, 50],
#     # "factor_arcs" : [1, 300],
# }


## For FIT
ranges_dict = {
    # "t": [130,300],
    # "tp": [-0.5,-0.2],
    # "tpp": [-0.14,0.14],
    # "tz": [-0.1,0.1],
    # "mu": [-1.8,-1.0],
    "gamma_0": [0.5,10],
    "gamma_k": [0,30],
    "power":[1, 100],
    # "gamma_dos_max": [0.1, 200],
    # "factor_arcs" : [1, 300],
}

## Data Tl2201 Tc = 20K (Hussey et al. 2003)  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
data_dict = {}  # keys (T, phi), content [filename, theta, rzz, theta_cut]
data_dict[4.2, 0] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_0.dat", 0, 1, 70]
data_dict[4.2, 20] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_20.dat", 0, 1, 70]
data_dict[4.2, 28] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_28.dat", 0, 1, 70]
data_dict[4.2, 36] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_36.dat", 0, 1, 70]
data_dict[4.2, 44] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_44.dat", 0, 1, 70]


# Play
# genetic_search(init_member,ranges_dict, data_dict, folder="sim/Tl2201_Tc_20K",
#                 population_size=5, N_generation=2, mutation_s=0.3, crossing_p=0.9)

# Play
# fit_search(init_member, ranges_dict, data_dict, folder="sim/Tl2201_Tc_20K")


# utils.save_member_to_json(init_member, folder="../data_NdLSCO_0p25")
# init_member = utils.load_member_from_json(
#     "sim/Tl2201_Tc_20K",
#     "data_p0.25_T4.2_fit_p0.250_T0_B45_t181.0_mu-1.480_tp-0.500_tpp0.200_tz0.015_tzz0.000_HolePocket_gzero6.4_gdos0.0_gk21.3_pwr39.5_arc1.0"
# )
utils.fig_compare(init_member, data_dict, folder="sim/Tl2201_Tc_20K")
