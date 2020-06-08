import unittest
from copy import deepcopy
import numpy as np
from cuprates_transport.bandstructure import BandStructure, Pocket, setMuToDoping, doping
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

class TestTransport(unittest.TestCase):

    ## ONE BAND Horio et al. //////
    params = {
            "bandname": "LargePocket",
            "a": 3.74767,
            "b": 3.74767,
            "c": 13.2,
            "t": 190,
            "tp": -0.14,
            "tpp": 0.07,
            "tz": 0.07,
            "mu": -0.826,
            "fixdoping": 0.24,
            "numberOfKz": 7,
            "mesh_ds": 1/20,
            "T" : 0,
            "Bamp": 45,
            "Btheta_min": 0,
            "Btheta_max": 90,
            "Btheta_step": 5,
            "Bphi_array": [0, 15, 30, 45],
            "gamma_0": 15.1,
            "gamma_k": 66,
            "gamma_dos_max": 0,
            "power": 12,
            "factor_arcs": 1,
        }

    def test_doping(self):
        bandObject = BandStructure(**TestTransport.params)
        bandObject.doping()
        self.assertEqual(np.round(bandObject.p,3), 0.239)

    def test_conductivity_T_0_B_0(self):
        """T = 0 & B = 0"""

        bandObject = BandStructure(**TestTransport.params)

        ## Discretize
        bandObject.doping()
        bandObject.discretize_FS()
        bandObject.dos_k_func()

        ## Conductivity
        condObject = Conductivity(bandObject, **TestTransport.params)
        condObject.Bamp = 0
        condObject.runTransport()
        condObject.chambersFunc(i=2, j=2)

        self.assertEqual(np.round(condObject.sigma[2,2],3), 18817.908)


    def test_conductivity_T_0(self):
        """T = 0"""

        bandObject = BandStructure(**TestTransport.params)

        ## Discretize
        bandObject.doping()
        bandObject.discretize_FS()
        bandObject.dos_k_func()

        ## Conductivity
        condObject = Conductivity(bandObject, **TestTransport.params)
        condObject.runTransport()
        condObject.chambersFunc(i=2, j=2)

        self.assertEqual(np.round(condObject.sigma[2,2],3), 18103.539)


    def test_conductivity_T(self):
        """T > 0"""

        params = deepcopy(TestTransport.params)
        params["T"] = 25 # in K

        bandObject = BandStructure(**params)

        ## Discretize
        bandObject.doping()
        bandObject.discretize_FS()
        bandObject.dos_k_func()

        ## Conductivity
        condObject = Conductivity(bandObject, **params)
        condObject.runTransport()
        condObject.chambersFunc(i=2, j=2, coeff_name="sigma")

        self.assertEqual(np.round(condObject.sigma[2,2],3), 17946.592)

if __name__ == '__main__':
    unittest.main()