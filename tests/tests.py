import unittest
from copy import deepcopy
import numpy as np
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.conductivity import Conductivity
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #


class TestTransport(unittest.TestCase):

    # ONE BAND Horio et al. //////
    params = {
            "band_name": "LargePocket",
            "a": 3.74767,
            "b": 3.74767,
            "c": 13.2,
            "energy_scale": 190,
            "band_params": {"mu": -0.826, "t": 1, "tp": -0.14, "tpp": 0.07,
                            "tz": 0.07},
            "fixdoping": 0.24,
            "res_xy": 20,
            "res_z": 7,
            "T": 0,
            "Bamp": 45,
            "Bphi_array": [0, 15, 30, 45],
            "gamma_0": 15.1,
            "gamma_k": 66,
            "power": 12,
        }

    def test_doping(self):
        bandObject = BandStructure(**TestTransport.params)
        bandObject.doping()
        self.assertEqual(np.round(bandObject.p, 3), 0.239)

    def test_conductivity_T_0_B_0(self):
        """T = 0 & B = 0"""

        bandObject = BandStructure(**TestTransport.params)

        # Discretize
        bandObject.runBandStructure()
        # bandObject.figMultipleFS2D()
        # bandObject.figDiscretizeFS2D()

        # Conductivity
        condObject = Conductivity(bandObject, **TestTransport.params)
        condObject.Bamp = 0
        condObject.runTransport()

        print(condObject.sigma)

        self.assertEqual(np.round(condObject.sigma[2, 2], 3), 25819.083)

    def test_conductivity_T_0(self):
        """T = 0"""

        bandObject = BandStructure(**TestTransport.params)

        # Discretize
        bandObject.runBandStructure()

        # Conductivity
        condObject = Conductivity(bandObject, **TestTransport.params)
        condObject.runTransport()

        self.assertEqual(np.round(condObject.sigma[2, 2], 3), 24999.56)

    def test_conductivity_T(self):
        """T > 0"""

        params = deepcopy(TestTransport.params)
        params["T"] = 25  # in K

        bandObject = BandStructure(**params)

        # Discretize
        bandObject.runBandStructure()

        # Conductivity
        condObject = Conductivity(bandObject, **params)
        condObject.runTransport()

        self.assertEqual(np.round(condObject.sigma[2, 2], 3), 24200.735)


if __name__ == '__main__':
    unittest.main()
