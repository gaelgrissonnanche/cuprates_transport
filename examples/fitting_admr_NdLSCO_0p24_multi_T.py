35.8e-5from numpy import deg2rad
from gael_modified.fitting_admr_multi_T import FittingADMR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

## ONE BAND Matt et al. ///////////////////////////////////////////////////////
init_member = {
    "bandname": "LargePocket",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "energy_scale": 190,
    "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
    "res_xy": 20,
    "res_z": 7,
    "fixdoping": 2,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": {6: 0, 12: 0, 20: 0, 25: 0},
    "Btheta_max": {6: 90, 12: 90, 20: 90, 25: 90},
    "Btheta_step": {6: 5, 12: 5, 20: 5, 25: 5},
    "Bphi_array": {6: [0, 45], 12: [0, 45], 20: [0, 45], 25: [0, 45]},
    "gamma_0": {6: 11, 12: 12, 20: 14, 25: 15},
    "gamma_k": {6: 70, 12: 70, 20: 70, 25: 70},
    "power": {6: 12, 12: 12, 20: 12, 25: 12},
    "data_T": [6, 12, 20, 25],
    "data_p": 0.24,
}

## For FIT
ranges_dict = {
    # "energy_scale": [170,210],
    # "tp": [-0.18,-0.1],
    # "tpp": [0.04,0.1],
    # "tz": [0.04,0.1],
    # "mu": [-1.0,-0.6],
    "gamma_0": {6: [9,12], 12: [11,14], 20: [13,15], 25: [14,16]},
    "gamma_k": {6: [50,100], 12: [50,100], 20: [50,100], 25: [50,100]},
    "power":   {6: [10,13], 12: [10,13], 20: [10,13], 25: [10,13]}
}


## Data Nd-LSCO 0.24  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
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

# ## Data Nd-LSCO 0.21  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, factor_to_SI]
# data_dict[25, 0] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_0.dat", 0, 2, 90, 35.8e-5]
# data_dict[25, 15] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_15.dat", 0, 2, 90, 35.8e-5]
# data_dict[25, 30] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_30.dat", 0, 2, 90, 35.8e-5]
# data_dict[25, 45] = ["data/NdLSCO_0p21/NdLSCO_0p21_1808A_c_AS_T_25_H_45_phi_45.dat", 0, 2, 90, 35.8e-5]



fitObject = FittingADMR(init_member, ranges_dict, data_dict, folder="sim/NdLSCO_0p24",
                        method="ampgo", normalized_data=False)
fitObject.weight_rhozz = 10

fitObject.runFit()
# fitObject.fig_compare()
