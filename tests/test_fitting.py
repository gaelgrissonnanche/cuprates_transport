import unittest
import numpy as np
from np import deg2rad
from cuprates_transport.fitting_admr import FittingADMR
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


class Tests_FittingADMR(unittest.TestCase):

    # ONE BAND Horio et al.
    init_memeber = {
        "band_name": "LargePocket",
        "a": 3.74767,
        "b": 3.74767,
        "c": 13.2,
        "energy_scale": 190,
        "band_params": {"mu": -0.826, "t": 1, "tp": -0.14, "tpp": 0.07,
                        "tz": 0.07},
        "fixdoping": 0.24,
        "res_xy": 21,
        "res_z": 7,
        "T": 0,
        "Bamp": 45,
        "Bphi_array": [0, 15],
        "gamma_0": 15.1,
        "gamma_k": 66,
        "gamma_dos_max": 0,
        "power": 12,
        "factor_arcs": 1,
        "data_T": 25,
        "data_p": 0.24,
        "epsilon_z": "",
    }

    ranges_dict = {
        "tz": [0.04, 0.1],
    }

    data_dict = {}
    prepath = "data/NdLSCO_0p24/0p25_%ddegr_45T_%dK.dat"
    data_dict[25, 0] = [prepath % (0, 25), 0, 1, 90, 6.71e-5]
    data_dict[25, 15] = [prepath % (15, 25), 0, 1, 90, 6.71e-5]

    def test_fit_least_square(self):
        """
        Test a simple and quick fit.
        """
        i_memb = Tests_FittingADMR.init_member
        r_dict = Tests_FittingADMR.ranges_dict
        d_dict = Tests_FittingADMR.data_dict

        fObj = FittingADMR(i_memb, r_dict, d_dict, folder="sim/NdLSCO_0p24",
                           method="least_square", normalized_data=True)

        fObj.runFit()
