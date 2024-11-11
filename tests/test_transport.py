import unittest
import numpy as np
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
from cuprates_transport.admr import ADMR
from copy import deepcopy
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


class Tests_BandStructure(unittest.TestCase):

    # ONE BAND Horio et al.
    params = {
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
            "Bphi_array": [0, 15, 30, 45],
            "scattering_params":{"constant": {"gamma_0":15.1},
                                 "cos2phi": {"gamma_k": 66, "power": 12}}
        }

    def test_initialization(self):
        """
        Test to make sure everything is initialized as expected.
        """
        p = Tests_BandStructure.params
        bObj = BandStructure(**p)

        self.assertEqual([bObj.a, bObj.b, bObj.c], [p["a"], p["b"], p["c"]])
        self.assertEqual(bObj.parallel, True)
        self.assertEqual(bObj._band_params, p["band_params"])
        self.assertEqual(bObj.band_name, p["band_name"])
        self.assertEqual(bObj.numberOfBZ, 1)

        # # e_xy_sym, e_z_sym, e_3D_sym
        # print(bObj.e_3D_sym)
        # print(bObj.v_sym)
        # print(bObj.epsilon_func)
        # print(bObj.v_func)

        self.assertEqual([bObj.res_xy, bObj.res_z], [p["res_xy"], p["res_z"]])
        self.assertEqual(bObj.march_square, False)

        self.assertEqual([bObj.kf, bObj.vf], [None, None])
        self.assertEqual([bObj.dkf, bObj.dks, bObj.dkz], [None, None, None])
        self.assertEqual([bObj.dos_k, bObj.dos_epsilon], [None, None])
        self.assertEqual([bObj.p, bObj.n], [None, None])

        self.assertEqual(bObj.number_of_points_per_kz_list, [])

    def test_run_nomarching(self):
        """
        Test the runBandStructure function.
        """
        bObj = BandStructure(**Tests_BandStructure.params)
        bObj.runBandStructure(epsilon=0)

        # Check that the doping was well initialized
        self.assertEqual(np.round(bObj.p, 3), 0.239)

        # Test the first elements of each of these
        self.assertEqual(np.around(bObj.kf[0][0], 3), -0.754)
        self.assertEqual(np.around(bObj.dkf[0], 4), 0.0043)
        self.assertEqual(np.around(bObj.vf[0][0], 3), -9847.585)
        self.assertEqual(np.around(bObj.dos_k[0], 3), 6.91828710418067e+29)

    def test_run_marching(self):
        """
        Test the runBandStructure function.
        """
        bObj = BandStructure(**Tests_BandStructure.params)
        bObj.march_square = True
        bObj.runBandStructure(epsilon=0)

        # Check that the doping was well initialized
        self.assertEqual(np.round(bObj.p, 3), 0.239)

        # Test the first elements of each of these
        self.assertEqual(np.around(bObj.kf[0][0], 3), 0.709)
        self.assertEqual(np.around(bObj.dkf[0], 4), 0.0064)
        self.assertEqual(np.around(bObj.vf[0][0], 3), 28028.457)
        self.assertEqual(np.around(bObj.dos_k[0], 3), 3.383176435137846e+29)

    def test_figures(self):
        """
        Test of the generation of figures.
        """
        bObj = BandStructure(**Tests_BandStructure.params)
        bObj.march_square = True
        bObj.runBandStructure(epsilon=0)
        bObj.figDiscretizeFS3D()
        # OG: The next function somehow requires that we used Marching square,
        # otherwise self.dks are not defined. Why? mc for marching square?
        bObj.mass_func()
        bObj.figMultipleFS2D()
        bObj.figDiscretizeFS2D()


class Tests_Conductivity(unittest.TestCase):

    # ONE BAND Horio et al.
    params = {
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
            "Bphi_array": [0, 15, 30, 45],
            "gamma_0": 15.1,
            "gamma_k": 66,
            "power": 12,
        }

    def test_initialization(self):
        """
        Test to make sure everything is initialized as expected.
        """
        p = Tests_BandStructure.params
        bObj = BandStructure(**p)
        bObj.runBandStructure(printDoping=False)

        cObj = Conductivity(bObj, **p)

    def test_conductivity_T0_B0(self):
        """
        Test at zero temperature with no field.
        """
        p = Tests_BandStructure.params
        bObj = BandStructure(**p)
        bObj.runBandStructure(printDoping=False)

        cObj = Conductivity(bObj, **p)
        cObj.Bamp = 0
        cObj.runTransport()

        self.assertEqual(np.round(cObj.sigma[2, 2], 3), 25819.083)

    def test_conductivity_T0(self):
        """
        Test at zero temperature with field.
        """
        p = Tests_BandStructure.params
        bObj = BandStructure(**p)
        bObj.runBandStructure(printDoping=False)

        cObj = Conductivity(bObj, **p)
        cObj.runTransport()

        self.assertEqual(np.round(cObj.sigma[2, 2], 3), 24999.56)

    def test_conductivity_T(self):
        """
        Test at finite temperature with field.
        """
        p = deepcopy(Tests_BandStructure.params)
        p["T"] = 25  # in Kelvin
        bObj = BandStructure(**p)
        bObj.runBandStructure(printDoping=False)

        cObj = Conductivity(bObj, **p)
        cObj.runTransport()

        self.assertEqual(np.round(cObj.sigma[2, 2], 3), 24200.735)

    def test_conductivity_T_plots(self):
        """
        Test at finite temperature with field.
        """
        p = deepcopy(Tests_BandStructure.params)
        p["T"] = 25  # in Kelvin
        bObj = BandStructure(**p)
        bObj.march_square = True
        bObj.runBandStructure(printDoping=False)

        cObj = Conductivity(bObj, **p)
        cObj.runTransport()

        cObj.figScatteringColor()
        cObj.omegac_tau_func()
        self.assertEqual(np.round(cObj.omegac_tau_k[0], 3), 21.929)

        cObj.figOnekft()
        cObj.figScatteringPhi(kz=0)

        rho = np.linalg.inv(cObj.sigma).transpose()
        self.assertEqual(np.around(rho[0, 0], 10), 2.223e-07)
        self.assertEqual(np.around(rho[0, 1], 10), 1.13e-08)
        self.assertEqual(np.around(rho[2, 2], 7), 4.58e-05)


class Tests_ADMR(unittest.TestCase):

    # ONE BAND Horio et al.
    params = {
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
            "Bphi_array": [0, 15, 30, 45],
            "gamma_0": 15.1,
            "gamma_k": 66,
            "power": 12,
        }

    def test_initialization(self):
        """
        Test to make sure everything is initialized as expected.
        """
        p = Tests_BandStructure.params
        bObj = BandStructure(**p)
        bObj.runBandStructure(printDoping=False)

        cObj = Conductivity(bObj, **p)
        cObj.runTransport()

        admr = ADMR([cObj], **p)
        admr.show_progress = False
        admr.runADMR()
        admr.figADMR()

if __name__ == '__main__':
    unittest.main()
