from cuprates_transport.fitting_admr import FittingADMR
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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
    "tz2": 0.014,
    "tz3": 0.015,
    "tz4": 0.008,
    "mu": -1.33,
    "fixdoping": 0.30,
    "numberOfKz": 5,
    "mesh_ds": 1 / 20,
    "N_time": 500,
    "T": 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 20, 28, 36, 44],
    "gamma_0": 3.77,
    "gamma_k": 0,
    "power": 6,
    "gamma_dos_max": 0,
    "factor_arcs": 1,
    "data_T": 4.2,
    "data_p": 0.30,
}


## For FIT
ranges_dict = {
    # "t": [130,300],
    # "tp": [-0.5,-0.2],
    # "tpp": [-0.14,0.14],
    # "tppp": [-0.1,0.1],
    # "tpppp": [-0.05,0.05],
    # "tz":  [0.005, 0.015],
    "tz2": [0.005, 0.015],
    "tz3": [0.005, 0.015],
    "tz4": [0.005, 0.015],
    # "mu": [-1.8,-1.0],
    "gamma_0": [2,6],
    # "gamma_k": [0,30],
    # "power":[1, 100],
    # "gamma_dos_max": [0.1, 200],
    # "factor_arcs" : [1, 300],
}

## Data Tl2201 Tc = 20K (Hussey et al. 2003)  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
data_dict = {}  # keys (T, phi), content [filename, col_theta, col_rzz, theta_cut, rhozz_0]
data_dict[4.2, 0] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_0.dat", 0, 1, 82, 0.023e-2]
data_dict[4.2, 20] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_20.dat", 0, 1, 82, 0.023e-2]
data_dict[4.2, 28] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_28.dat", 0, 1, 82, 0.023e-2]
data_dict[4.2, 36] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_36.dat", 0, 1, 82, 0.023e-2]
data_dict[4.2, 44] = ["data/Tl2201_Tc_20K_Hussey_2003/rhozz_vs_theta_Tl2201_Tc_20K_B45_T4.2K_phi_44.dat", 0, 1, 82, 0.023e-2]

fitObject = FittingADMR(init_member, ranges_dict, data_dict, folder="sim/Tl2201_Tc_20K",
                        method="differential_evolution")
fitObject.runFit()
# fitObject.fig_compare()
