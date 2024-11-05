import numpy as np
from numpy import ones
from tqdm import tqdm
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

class ADMR:
    def __init__(self, condObject_list, Btheta_min=0, Btheta_max=110,
                 Btheta_step=5, Bphi_array=[0, 15, 30, 45], show_progress=True, **trash):
        # Band dictionary
        self.condObject_dict = {} # will contain the condObject for each band, with key their bandname
        self.total_filling = 0 # total bands filling (of electron) over all bands
        for condObject in condObject_list:
            self.total_filling += condObject.bandObject.n
            self.condObject_dict[condObject.bandObject.band_name] = condObject
        self.band_names = list(self.condObject_dict.keys())
        self.total_hole_doping = 1 - self.total_filling # total bands hole doping over all bands

        ## Miscellaneous
        self.show_progress = show_progress # shows progress bar or not

        # Magnetic field
        self.Btheta_min   = Btheta_min    # in degrees
        self.Btheta_max   = Btheta_max    # in degrees
        self.Btheta_step  = Btheta_step  # in degrees
        self.Btheta_array = np.arange(self.Btheta_min, self.Btheta_max + self.Btheta_step, self.Btheta_step)
        self.Bphi_array = np.array(Bphi_array)

        # Resistivity array rho_zz
        self.rhozz_array = None
        # Resistivity array rho_zz / rho_zz(0)
        self.rzz_array = None


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def runADMR(self):
        rhozz_array = np.empty((self.Bphi_array.size, self.Btheta_array.size), dtype= np.float64)
        if self.show_progress is True:
            iterator = enumerate(tqdm(self.Bphi_array, ncols=80, unit="phi", desc="ADMR"))
        else:
            iterator = enumerate(self.Bphi_array)
        for l, phi in iterator:
            for m, theta in enumerate(self.Btheta_array):
                sigma_tensor = 0
                for _, condObject in list(self.condObject_dict.items()):
                    condObject.Bphi = phi
                    condObject.Btheta = theta
                    condObject.runTransport()
                    sigma_tensor += condObject.sigma
                rho = np.linalg.inv(sigma_tensor)
                rhozz_array[l, m] = rho[2,2]
        rhozz_0_array = np.outer(rhozz_array[:, 0], np.ones(self.Btheta_array.shape[0]))
        self.rhozz_array = rhozz_array
        self.rzz_array = rhozz_array / rhozz_0_array
