import numpy as np
from numpy import cos, sin, pi, exp, sqrt, arctan2, cosh, arccosh
from scipy.integrate import odeint
from scipy.constants import Boltzmann, hbar, elementary_charge, physical_constants
from skimage import measure
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from copy import deepcopy
from cuprates_transport.scattering import Scattering
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Units ////////
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 1e-12 # 1 ps in seconds

## Constant //////
e = elementary_charge # C
kB = Boltzmann # J / K
kB = kB / meV # meV / K

## This coefficient takes into accound all units and constant to prefactor Chambers formula
units_chambers = 2 * e**2 / (2*pi)**3 * picosecond / Angstrom**2

class Conductivity(Scattering):
    def __init__(self, bandObject, Bamp, Bphi=0, Btheta=0,
                 N_time=500, rtol = 1e-4, atol = 1e-4,
                 T=0, dfdE_cut_percent=0.001, N_epsilon=20,
                 **kwargs):
        super().__init__(**kwargs)

        # Band object
        self.bandObject = deepcopy(bandObject)

        # Magnetic field in degrees
        self._Bamp   = Bamp # in Tesla
        self._Btheta = Btheta
        self._Bphi   = Bphi
        self._B_vector = self.B_func() # np array fo Bx,By,Bz
        self.omegac_tau = None

        # Temperature and energy integration
        self._T = T # in Kelvin
        self._N_epsilon = N_epsilon
        self._dfdE_cut_percent = dfdE_cut_percent
        if self._T != 0:
            self.epsilon_array = np.linspace(-self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                              self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                              self._N_epsilon)

        # Time parameters
        self._N_time = N_time # number of steps in time
        self.time_max = None # # in picoseconds
        self.dtime = None # in picoseconds
        self.time_array = None # in picoseconds
        self.dtime_array = None # in picoseconds

        # Maximum and Minimum scattering rate values
        self.gamma_tot_max = None
        self.gamma_tot_min = None

        ## Precision differential equation solver
        self.rtol = rtol # default is 1.49012e-8
        self.atol = atol # default is 1.49012e-8

        # Time-dependent kf, vf
        self.kft = np.empty(1)
        self.vft = np.empty(1)
        self.tau_tot = np.empty(1) # array[i0, i_t] with i0 index of the initial index
        self.t_o_tau = np.empty(1) # array[i0, i_t] with i0 index of the initial index
        # i_t index of the time from the starting position

        # Product of [vf x int(vft*exp(-t/tau))]
        self.v_product = np.empty(1)

        # Electric tensor: x, y, z = 0, 1, 2
        self.sigma = np.empty((3,3), dtype= np.float64)
        # Thermoelectric tensor: x, y, z = 0, 1, 2
        self.alpha = np.empty((3,3), dtype= np.float64)
        # Thermal tensor: x, y, z = 0, 1, 2
        self.beta = np.empty((3,3), dtype= np.float64)



    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def _get_B_vector(self):
        return self._B_vector
    def _set_B_vector(self, B_vector):
        print("Cannot access B_vector directly, just change Bamp, Bphi, Btheta")
    B_vector = property(_get_B_vector, _set_B_vector)

    def _get_Bamp(self):
        return self._Bamp
    def _set_Bamp(self, Bamp):
        self._Bamp = Bamp
        self._B_vector = self.B_func()
    Bamp = property(_get_Bamp, _set_Bamp)

    def _get_Bphi(self):
        return self._Bphi
    def _set_Bphi(self, Bphi):
        self._Bphi = Bphi
        self._B_vector = self.B_func()
    Bphi = property(_get_Bphi, _set_Bphi)

    def _get_Btheta(self):
        return self._Btheta
    def _set_Btheta(self, Btheta):
        self._Btheta = Btheta
        self._B_vector = self.B_func()
    Btheta = property(_get_Btheta, _set_Btheta)

    def _get_N_time(self):
        return self._N_time
    def _set_N_time(self, N_time):
        self._N_time = N_time
        self.dtime = self.time_max / self._N_time
        self.time_array = np.arange(0, self.time_max, self.dtime)
        # integrand for tau_function
        self.dtime_array = np.append(0, self.dtime * np.ones_like(self.time_array))[:-1]
    N_time = property(_get_N_time, _set_N_time)

    def _get_T(self):
        return self._T
    def _set_T(self, T):
        self._T = T
        if self._T != 0:
            bound = self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0)))
            self.epsilon_array = np.linspace(-bound,bound, self._N_epsilon)
    T = property(_get_T, _set_T)

    def _get_N_epsilon(self):
        return self._N_epsilon
    def _set_N_epsilon(self, N_epsilon):
        self._N_epsilon = N_epsilon
        bound = self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0)))
        self.epsilon_array = np.linspace(-bound,bound, self._N_epsilon)
    N_epsilon = property(_get_N_epsilon, _set_N_epsilon)

    def _get_dfdE_cut_percent(self):
        return self._dfdE_cut_percent
    def _set_dfdE_cut_percent(self, dfdE_cut_percent):
        self._dfdE_cut_percent  = dfdE_cut_percent
        bound = self.energyCutOff(dfdE_cut_percent * np.abs(self.dfdE(0)))
        self.epsilon_array = np.linspace(-bound,bound, self._N_epsilon)
    dfdE_cut_percent = property(_get_dfdE_cut_percent, _set_dfdE_cut_percent)


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def runTransport(self):
        self.reset_time() # calculates the upper cut of for the movement equation
        self.phi = None # reset the phi angle
        if self._T != 0:
            #!!!! Add a warning if bandObject not already discretized
            self.dos_k_epsilon      = {}
            self.dkf_epsilon        = {}
            self.kft_epsilon        = {}
            self.vft_epsilon        = {}
            self.t_o_tau_epsilon    = {}
            for epsilon in self.epsilon_array:
                self.bandObject.runBandStructure(epsilon = epsilon, printDoping=False)
                self.solve_movement()
                self.t_o_tau_func(epsilon)
                self.dos_k_epsilon[epsilon]      = self.bandObject.dos_k
                self.dkf_epsilon[epsilon]        = self.bandObject.dkf
                self.kft_epsilon[epsilon]        = self.kft
                self.vft_epsilon[epsilon]        = self.vft
                self.t_o_tau_epsilon[epsilon]    = self.t_o_tau
                ## !!!!  Do not forget to update scattering rates !!! ##
                ## Create properties for tmax, etc.
            self.bandObject.runBandStructure(epsilon = 0, printDoping=False)
            # this last one is to be sure the bandObject is at the FS at the end
            self.chambers_func()
        else:
            self.solve_movement()
            self.t_o_tau_func()
            self.chambers_func()

        self.gamma_tot_max = 1 / self.tau_total_min() # in THz
        self.gamma_tot_min = 1 / self.tau_total_max() # in THz

    def reset_time(self):
        # Time parameters
        self.time_max = 8 * self.tau_total_max()  # in picoseconds
        self.dtime = self.time_max / self.N_time
        self.time_array = np.arange(0, self.time_max, self.dtime)
        self.dtime_array = np.append(0, self.dtime * np.ones_like(self.time_array))[:-1] # integrand for tau_function

    def B_func(self):
        B = self._Bamp * np.array([sin(self._Btheta*pi/180) * cos(self._Bphi*pi/180),
                                   sin(self._Btheta*pi/180) * sin(self._Bphi*pi/180),
                                   cos(self._Btheta*pi/180)])
        return B

    def crossProductVectorized(self, vx, vy, vz):
        # (- B) represents -t in vj(-t, k) in the Chambers formula
        # if integrated from 0 to +infinity, instead of -infinity to 0
        product_x = vy[:] * -self._B_vector[2] - vz[:] * -self._B_vector[1]
        product_y = vz[:] * -self._B_vector[0] - vx[:] * -self._B_vector[2]
        product_z = vx[:] * -self._B_vector[1] - vy[:] * -self._B_vector[0]
        return np.vstack((product_x, product_y, product_z))

    def solve_movement(self):
        len_t = self.time_array.shape[0]
        len_kf = self.bandObject.kf.shape[1]
        ## Magnetic Field ON
        if self.Bamp != 0:
            # Flatten to get all the initial kf solved at the same time
            kf0 = self.bandObject.kf.flatten()
            # Sovle differential equation
            self.kft = odeint(self.movement_equation, kf0, self.time_array,
                              rtol = self.rtol, atol = self.atol).transpose()
            # Reshape arrays
            self.kft.shape = (3, len_kf, len_t)
            # Velocity function of time
            self.vft = np.empty_like(self.kft, dtype = np.float64)
            kft0, kft1, kft2 = self.kft[0, :, :], self.kft[1, :, :], self.kft[2, :, :]
            vft0, vft1, vft2 = self.bandObject.velocity_func(kft0, kft1, kft2)
            self.vft[0, :, :], self.vft[1, :, :], self.vft[2, :, :] = vft0, vft1, vft2
        ## Magnetic Field OFF
        else:
            self.kft = np.empty((3, len_kf, 1), dtype = np.float64)
            self.vft = np.empty((3, len_kf, 1), dtype = np.float64)
            kf0, kf1, kf2 = (self.bandObject.kf[0, :], self.bandObject.kf[1, :],
                             self.bandObject.kf[2, :])
            self.kft[0, :, 0], self.kft[1, :, 0], self.kft[2, :, 0] = kf0, kf1, kf2
            vf0, vf1, vf2 = (self.bandObject.vf[0, :], self.bandObject.vf[1, :],
                             self.bandObject.vf[2, :])
            self.vft[0, :, 0], self.vft[1, :, 0], self.vft[2, :, 0] = vf0, vf1, vf2

    def movement_equation(self, k, t):
        len_k = int(k.shape[0]/3)
        k.shape = (3, len_k) # reshape the flatten k
        vx, vy, vz =  self.bandObject.velocity_func(k[0,:], k[1,:], k[2,:])
        dkdt = ( - e * picosecond * Angstrom / hbar ) * self.crossProductVectorized(vx, vy, vz)
        dkdt.shape = (3*len_k,) # flatten k again
        return dkdt


    ## Chambers formula's parts -------------------------------------------------#

    def velocity_product(self, kft, vft, t_o_tau):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0: vif = vxf """
        if self.Bamp != 0:
            vi = vft[:, :, 0] # (3, len_kf)
            vj = np.sum(vft[:, :, :] * exp(-t_o_tau) * self.dtime, axis=2) # (3, len_kf)
            self.v_product = np.einsum('ji,ki->jki',vi,vj) # (3, 3, len_kf)
                                        # with 3, 3 the indices for i, j in sigma
        else:
            self.v_product = np.einsum('ji,ki->jki', vft[:, :, 0], vft[:, :, 0] * (1 / t_o_tau))
        return self.v_product

    def sigma_epsilon(self, dos_k, dkf, kft, vft, t_o_tau):
        sigma_epsilon = (units_chambers / self.bandObject.number_of_bz *
         np.sum(dkf * dos_k * self.velocity_product(kft, vft, t_o_tau), axis=2))
        return sigma_epsilon

    def integrand_coeff(self, epsilon, coeff_name):
        if coeff_name == "sigma":
            return 1
        elif coeff_name == "alpha":
            return (epsilon * meV) / self.T / (-e)
        elif coeff_name == "beta":
            return (epsilon * meV)**2 / self.T / (-e)**2
        else:
            print("!Warming! You have not enter a correct coefficient name")

    def chambers_func(self):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0 and j = 1 : sigma[i,j] = sigma_xy """
        if self._T == 0:
            coeff_tot = self.sigma_epsilon(self.bandObject.dos_k,
                                                  self.bandObject.dkf,
                                                  self.kft, self.vft,
                                                  self.t_o_tau)
            self.sigma = coeff_tot
        else:
            sigma_tot = 0
            alpha_tot = 0
            beta_tot = 0
            d_epsilon = self.epsilon_array[1] - self.epsilon_array[0]
            for epsilon in self.epsilon_array:
                sigma_epsilon = self.sigma_epsilon(self.dos_k_epsilon[epsilon],
                                                   self.dkf_epsilon[epsilon],
                                                   self.kft_epsilon[epsilon],
                                                   self.vft_epsilon[epsilon],
                                                   self.t_o_tau_epsilon[epsilon])
                # Sum over the energie
                sigma_tot += d_epsilon * (- self.dfdE(epsilon)) * \
                             self.integrand_coeff(epsilon, "sigma") * sigma_epsilon
                alpha_tot += d_epsilon * (- self.dfdE(epsilon)) * \
                             self.integrand_coeff(epsilon, "alpha") * sigma_epsilon
                beta_tot  += d_epsilon * (- self.dfdE(epsilon)) * \
                             self.integrand_coeff(epsilon, "beta") * sigma_epsilon
            self.sigma = sigma_tot
            self.alpha = alpha_tot
            self.beta  = beta_tot

    def dfdE(self, epsilon):
        if self._T == 0:
            return 1
        """Returns in fact dfdE * t in order to get epsilon unitless"""
        return -1 / (4 * kB * self._T) / (cosh(epsilon / (2*kB * self._T)))**2

    def energyCutOff(self, dfdE_cut):
        if self._T != 0:
            return 2*kB*self._T * arccosh(1/sqrt(dfdE_cut * 4*kB*self._T))
        else:
            return 0

    def omegac_tau_func(self):
        dks = self.bandObject.dks / Angstrom # in m^-1
        kf = self.bandObject.kf
        vf = self.bandObject.vf # in m / s
        kf_perp = sqrt(kf[0, :]**2 + kf[1, :]**2) / Angstrom  # kf perp to B in m
        vf_perp = sqrt(vf[0, :]**2 + vf[1, :]**2)  # vf perp to B, in m / s
        prefactor = hbar / (2 * pi * e * self._Bamp)
        ## Function of k
        kf0, kf1, kf2 = kf[0, :], kf[1, :], kf[2, :]
        vf0, vf1, vf2 = vf[0, :], vf[1, :], kf[2, :]
        inverse_omegac_tau_k = (prefactor * 2*pi*kf_perp / vf_perp / picosecond *
                                self.tau_tot_func(kf0, kf1, kf2, vf0, vf1, vf2))
        self.omegac_tau_k = 1 / inverse_omegac_tau_k
        # ## Integrated over the Fermi surface
        # inverse_omegac_tau = np.sum(prefactor * dks / vf_perp / (picosecond * self.tau_tot_func(kf[0, :], kf[1, :], kf[2, :], vf[0, :], vf[1, :], vf[2, :]))) / self.bandObject.res_z  # divide by the number of kz to average over all kz
        # self.omegac_tau = 1 / inverse_omegac_tau

    ## Bring up the total scattering time \tau ----------------------------------#
    def t_o_tau_func(self, epsilon = 0):
        ## Integral from 0 to t of dt' / tau( k(t') ) or dt' * gamma( k(t') )
        ## Magnetic Field ON
        if self.Bamp !=0:
            self.tau_tot = self.tau_tot_func(self.kft[0, :, :],
                                             self.kft[1, :, :],
                                             self.kft[2, :, :],
                                             self.vft[0, :, :],
                                             self.vft[1, :, :],
                                             self.vft[2, :, :],
                                             epsilon)
            self.t_o_tau = np.cumsum( self.dtime_array / self.tau_tot, axis = 1)
        ## Magnetic Field OFF
        else:
            self.tau_tot = self.tau_tot_func(self.kft[0, :, 0],
                                             self.kft[1, :, 0],
                                             self.kft[2, :, 0],
                                             self.vft[0, :, 0],
                                             self.vft[1, :, 0],
                                             self.vft[2, :, 0],
                                             epsilon)
            self.t_o_tau = 1 / self.tau_tot

    def tau_total_max(self):
        # Compute the tau_max (the longest time between two collisions)
        # to better integrate from 0 --> 8 * 1 / gamma_min (instead of infinity)
        kf = self.bandObject.kf
        vf = self.bandObject.vf
        return np.max(self.tau_tot_func(kf[0, :], kf[1, :], kf[2, :],
                                          vf[0, :], vf[1, :], vf[2, :]))
    def tau_total_min(self):
        kf = self.bandObject.kf
        vf = self.bandObject.vf
        return np.min(self.tau_tot_func(kf[0, :], kf[1, :], kf[2, :],
                                          vf[0, :], vf[1, :], vf[2, :]))




    ## Plotting -----------------------------------------------------------------#


    #///// RC Parameters //////#
    mpl.rcdefaults()
    mpl.rcParams['font.size'] = 24. # change the size of the font in every figure
    mpl.rcParams['font.family'] = 'Arial' # font Arial in every figure
    mpl.rcParams['axes.labelsize'] = 24.
    mpl.rcParams['xtick.labelsize'] = 24
    mpl.rcParams['ytick.labelsize'] = 24
    mpl.rcParams['xtick.direction'] = "in"
    mpl.rcParams['ytick.direction'] = "in"
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.major.width'] = 0.6
    mpl.rcParams['ytick.major.width'] = 0.6
    mpl.rcParams['axes.linewidth'] = 0.6 # thickness of the axes lines
    mpl.rcParams['pdf.fonttype'] = 3  # Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
    # editing the text in illustrator


    def figScatteringColor(self, kz=0, gamma_min=None, gamma_max=None, mesh_xy=501):
        bObj = self.bandObject
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 5.6))
        fig.subplots_adjust(left=0.10, right=0.85, bottom=0.20, top=0.9)

        kx_a = np.linspace(-pi / bObj.a, pi / bObj.a, mesh_xy)
        ky_a = np.linspace(-pi / bObj.b, pi / bObj.b, mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = bObj.energy_func(kxx, kyy, kz)
        contours = measure.find_contours(bands, 0)

        gamma_max_list = []
        gamma_min_list = []
        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / bObj.a
            ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / bObj.b
            vx, vy, vz = bObj.velocity_func(kx, ky, kz)

            gamma_kz = 1 / self.tau_tot_func(kx, ky, kz, vx, vy, vz)
            gamma_max_list.append(np.max(gamma_kz))
            gamma_min_list.append(np.min(gamma_kz))

            points = np.array([kx*bObj.a, ky*bObj.b]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=plt.get_cmap('gnuplot'))
            lc.set_array(gamma_kz)
            lc.set_linewidth(4)

            line = axes.add_collection(lc)

        if gamma_min == None:
            gamma_min = min(gamma_min_list)
        if gamma_max == None:
            gamma_min = min(gamma_min_list)
        line.set_clim(gamma_min, gamma_max)
        cbar = fig.colorbar(line, ax=axes)
        cbar.minorticks_on()
        cbar.ax.set_ylabel(r'$\Gamma_{\rm tot}$ ( THz )', rotation=270, labelpad=40)

        kz = np.round(kz/(np.pi/bObj.c), 1)
        fig.text(0.97, 0.05, r"$k_{\rm z}$ = " + "{0:g}".format(kz) + r"$\pi$/c",
                ha="right", color="r")
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$k_{\rm x}$", labelpad=8)
        axes.set_ylabel(r"$k_{\rm y}$", labelpad=8)

        axes.set_xticks([-pi, 0., pi])
        axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axes.set_yticks([-pi, 0., pi])
        axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        plt.show()


    def figScatteringPhi(self, kz=0, mesh_xy=501):
        bObj = self.bandObject
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 4.6))
        fig.subplots_adjust(left=0.20, right=0.8, bottom=0.20, top=0.9)
        axes2 = axes.twinx()
        axes2.set_axisbelow(True)

        ###
        kx_a = np.linspace(-pi / bObj.a, pi / bObj.a, mesh_xy)
        ky_a = np.linspace(-pi / bObj.b, pi / bObj.b, mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = bObj.energy_func(kxx, kyy, kz)
        contours = measure.find_contours(bands, 0)

        for contour in contours:
            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / bObj.a
            ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / bObj.b
            vx, vy, vz = bObj.velocity_func(kx, ky, kz)

            gamma_kz = 1 / self.tau_tot_func(kx, ky, kz, vx, vy, vz)

            phi = np.rad2deg(np.arctan2(ky,kx))

            line = axes2.plot(phi, vz)
            plt.setp(line, ls ="", c = '#80ff80', lw = 3, marker = "o",
                    mfc = '#80ff80', ms = 3, mec = "#7E2320", mew= 0)  # set properties

            line = axes.plot(phi, gamma_kz)
            plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k',
                    ms = 3, mec = "#7E2320", mew= 0)  # set properties

        axes.set_xlim(0, 180)
        axes.set_xticks([0, 45, 90, 135, 180])
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$\phi$", labelpad = 8)
        axes.set_ylabel(r"$\Gamma_{\rm tot}$ ( THz )", labelpad=8)
        axes2.set_ylabel(r"$v_{\rm z}$", rotation = 270, labelpad =25, color="#80ff80")

        kz = np.round(kz/(np.pi/bObj.c), 1)
        fig.text(0.97, 0.05, r"$k_{\rm z}$ = " + "{0:g}".format(kz) + r"$\pi$/c",
                ha="right", color="r", fontsize = 20)
        # axes.tick_params(axis='x', which='major', pad=7)
        # axes.tick_params(axis='y', which='major', pad=8)
        # axes.set_xlabel(r"$k_{\rm x}$", labelpad=8)
        # axes.set_ylabel(r"$k_{\rm y}$", labelpad=8)
        # axes.set_xticks([-pi, 0., pi])
        # axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        # axes.set_yticks([-pi, 0., pi])
        # axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        plt.show()


    def figOnekft(self, index_kf = 0, meshXY = 1001):
        mesh_graph = meshXY
        bObj = self.bandObject
        kx = np.linspace(-pi / bObj.a, pi / bObj.a, mesh_graph)
        ky = np.linspace(-pi / bObj.b, pi / bObj.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')
        fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6))
        fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91)
        fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)
        line = axes.contour(kxx*bObj.a, kyy*bObj.b,
                            bObj.energy_func(kxx, kyy, - 2*pi / bObj.c), 0, colors = '#FF0000', linewidths = 3)
        line = axes.plot(self.kft[0, index_kf,:]*bObj.a, self.kft[1, index_kf,:]*bObj.b)
        plt.setp(line, ls ="-", c = 'b', lw = 1, marker = "", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0) # trajectory
        line = axes.plot(bObj.kf[0, index_kf]*bObj.a, bObj.kf[1, index_kf]*bObj.b)
        plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 4.5, mec = "b", mew= 1.5)  # starting point
        line = axes.plot(self.kft[0, index_kf, -1]*bObj.a, self.kft[1, index_kf, -1]*bObj.b)
        plt.setp(line, ls ="", c = 'b', lw = 1, marker = "o", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0)  # end point
        axes.set_xlim(-pi, pi)
        axes.set_ylim(-pi, pi)
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
        axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)
        axes.set_xticks([-pi, 0., pi])
        axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axes.set_yticks([-pi, 0., pi])
        axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        plt.show()


    def figOnevft(self, index_kf = 0):
        fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
        fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95)
        axes.axhline(y = 0, ls ="--", c ="k", linewidth = 0.6)
        line = axes.plot(self.time_array, self.vft[2, index_kf,:])
        plt.setp(line, ls ="-", c = '#6AFF98', lw = 3, marker = "", mfc = '#6AFF98', ms = 5, mec = "#7E2320", mew= 0)
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$t$", labelpad = 8)
        axes.set_ylabel(r"$v_{\rm z}$", labelpad = 8)
        axes.locator_params(axis = 'y', nbins = 6)
        plt.show()

    def figCumulativevft(self, index_kf = -1):
        fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
        fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95)
        line = axes.plot(self.time_array, np.cumsum(self.vft[2, index_kf, :] * exp(-self.t_o_tau[index_kf, :])))
        plt.setp(line, ls ="-", c = 'k', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$t$", labelpad = 8)
        axes.set_ylabel(r"$\sum_{\rm t}$ $v_{\rm z}(t)$$e^{\rm \dfrac{-t}{\tau}}$", labelpad = 8)
        axes.locator_params(axis = 'y', nbins = 6)
        plt.show()


    def figdfdE(self):
        fig, axes = plt.subplots(1, 1, figsize = (9.2, 5.6))
        fig.subplots_adjust(left = 0.17, right = 0.81, bottom = 0.18, top = 0.95)
        axes.axhline(y=0, ls="--", c="k", linewidth=0.6)
        dfdE_cut    = self._dfdE_cut_percent * np.abs(self.dfdE(0))
        epsilon_cut = self.energyCutOff(dfdE_cut)
        d_epsilon   = self.epsilon_array[1]-self.epsilon_array[0]
        epsilon = np.arange(-epsilon_cut-10*d_epsilon, epsilon_cut+11*d_epsilon, d_epsilon)
        dfdE = self.dfdE(epsilon)
        line = axes.plot(epsilon, -dfdE)
        plt.setp(line, ls ="-", c = 'r', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties
        ## Cuts
        line = axes.plot(epsilon_cut, dfdE_cut)
        plt.setp(line, ls ="", c = 'k', lw = 3, marker = "s", mfc = 'k', ms = 6, mec = "#7E2320", mew= 0)  # set properties
        line = axes.plot(-epsilon_cut, dfdE_cut)
        plt.setp(line, ls ="", c = 'k', lw = 3, marker = "s", mfc = 'k', ms = 6, mec = "#7E2320", mew= 0)  # set properties
        ## Labels
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$\epsilon$", labelpad = 8)
        axes.set_ylabel(r"-$df/d\epsilon$ ( units of t )", labelpad = 8)
        axes.locator_params(axis = 'y', nbins = 6)
        plt.show()

    def figParameters(self, fig_show=True):
        """
        Figure that regroups the parameters of the computation
        """
        ## Dictionary of labels + parameters
        labels_dict = {}
        labels_dict["band name"] = self.bandObject.band_name
        labels_dict["a"] = self.bandObject.a
        labels_dict["b"] = self.bandObject.b
        labels_dict["c"] = self.bandObject.c
        labels_dict["energy scale"] = self.bandObject.energy_scale
        for label, value in self.bandObject._band_params.items():
            labels_dict[label] = value
        labels_dict["# of BZ"] = self.bandObject.number_of_bz
        for param, value in self._scattering_params.items():
            labels_dict[param] = value

        ## Subfigure that serves the table of parameters
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
        fig.subplots_adjust(left=0.15, right=0.25, bottom=0.18, top=0.95)
        axes.remove()
        axes_params = plt.axes([0.8, 0.12, .4, .4])
        # Hide the axes
        axes_params.axis('off')
        # Convert dictionary into rows for the table
        rows = [[key, value] for key, value in labels_dict.items()]
        # Create a table
        table = axes_params.table(
            cellText=rows,
            colLabels=["Parameter", "Value"],
            loc="left",
            cellLoc="left",
            colColours=["#fd9d7a", "#fd9d7a"]
        )
        # Increase row height
        row_height = 0.09
        for i, cell in table.get_celld().items():
            cell.set_height(row_height)
        # Adjust table properties
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.auto_set_column_width(col=list(range(len(rows[0]))))  # Auto-adjust column width

        ## Insets >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
        ## Meshgrid
        a = self.bandObject.a
        b = self.bandObject.b
        c = self.bandObject.c
        mesh_graph = 201
        kx = np.linspace(-pi/a, pi/a, mesh_graph)
        ky = np.linspace(-pi/b, pi/b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
        ## Inset FS /////////////////////////////////////////////////////////////#
        axes_FS = plt.axes([0, 0.56, .4, .4])
        axes_FS.contour(kxx*a, kyy*b, self.bandObject.energy_func(
            kxx, kyy, 0), 0, colors='#ff0000', linewidths=1)
        axes_FS.contour(kxx*a, kyy*b, self.bandObject.energy_func(
            kxx, kyy, pi / c), 0, colors='#ee9c54', linewidths=1)
        axes_FS.contour(kxx*a, kyy*b, self.bandObject.energy_func(
            kxx, kyy, 2 * pi / c), 0, colors='#f3d36b', linewidths=1)
        fig.text(0.33, 0.67, r"$k_{\rm z}$", fontsize=14)
        fig.text(0.33, 0.63, r"0", color='#ff0000', fontsize=14)
        fig.text(0.33, 0.60, r"$\pi$/c", color='#ee9c54', fontsize=14)
        fig.text(0.33, 0.57, r"$2\pi$/c", color='#f3d36b', fontsize=14)
        ## Axis labels
        axes_FS.set_xlabel(r"$k_{\rm x}$", labelpad=0, fontsize=14)
        axes_FS.set_ylabel(r"$k_{\rm y}$", labelpad=-5, fontsize=14)
        ## Ticks labels
        a_x = np.round(self.bandObject.k_max[0] / pi, 1)
        axes_FS.set_xlabel(r"$k_{\rm x}$", labelpad = 8, fontsize=14)
        axes_FS.set_xticks([-self.bandObject.k_max[0], 0., self.bandObject.k_max[0]])
        axes_FS.set_xticklabels([r"-" + f"{a_x:g}" + r"$\pi$", "0", f"{a_x:g}" + r"$\pi$"], fontsize=14)
        a_y = np.round(self.bandObject.k_max[1] / pi, 1)
        axes_FS.set_ylabel(r"$k_{\rm y}$", labelpad = 8, fontsize=14)
        axes_FS.set_yticks([-self.bandObject.k_max[1], 0., self.bandObject.k_max[1]])
        axes_FS.set_yticklabels([r"-" + f"{a_y:g}" + r"$\pi$", "0", f"{a_y:g}" + r"$\pi$"], fontsize=14)
        axes_FS.set_aspect(aspect=1)

        ## Inset Scattering rate ////////////////////////////////////////////////#
        axes_srate = plt.axes([-0.01, 0.04, .4, .4])
        bands = self.bandObject.energy_func(kxx, kyy, 0)
        contours = measure.find_contours(bands, 0)
        gamma_max_list = []
        gamma_min_list = []
        for contour in contours:
            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_graph - 1) - 0.5) * 2 * pi / a
            ky = (contour[:, 1] / (mesh_graph - 1) - 0.5) * 2 * pi / b
            vx, vy, vz = self.bandObject.velocity_func(kx, ky, np.zeros_like(kx))
            gamma_kz0 = 1 / self.tau_tot_func(kx, ky, 0, vx, vy, vz)
            gamma_max_list.append(np.max(gamma_kz0))
            gamma_min_list.append(np.min(gamma_kz0))
            points = np.array([kx * a, ky * b]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=plt.get_cmap('gnuplot'))
            lc.set_array(gamma_kz0)
            lc.set_linewidth(4)
            line = axes_srate.add_collection(lc)
        gamma_max = max(gamma_max_list)
        gamma_min = min(gamma_min_list)
        line.set_clim(gamma_min, gamma_max)
        cbar = fig.colorbar(line, ax=axes_srate)
        cbar.minorticks_on()
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_ylabel(r'$\Gamma_{\rm tot}$ ( THz )',
                        rotation=270,
                        labelpad=20,
                        fontsize=14)
        fig.text(0.305, 0.405, r"$k_{\rm z}$=0", fontsize=10, ha="right")
        axes_srate.tick_params(axis='x', which='major')
        axes_srate.tick_params(axis='y', which='major')
        # axes_srate.set_xlabel(r"$k_{\rm x}$", fontsize=14)
        axes_srate.set_ylabel(r"$k_{\rm y}$", fontsize=14, labelpad=-5)
        ## Ticks labels
        a_x = np.round(self.bandObject.k_max[0] / pi, 1)
        axes_srate.set_xlabel(r"$k_{\rm x}$", labelpad = 8, fontsize=14)
        axes_srate.set_xticks([-self.bandObject.k_max[0], 0., self.bandObject.k_max[0]])
        axes_srate.set_xticklabels([r"-" + f"{a_x:g}" + r"$\pi$", "0", f"{a_x:g}" + r"$\pi$"], fontsize=14)
        a_y = np.round(self.bandObject.k_max[1] / pi, 1)
        axes_srate.set_ylabel(r"$k_{\rm y}$", labelpad = 8, fontsize=14)
        axes_srate.set_yticks([-self.bandObject.k_max[1], 0., self.bandObject.k_max[1]])
        axes_srate.set_yticklabels([r"-" + f"{a_y:g}" + r"$\pi$", "0", f"{a_y:g}" + r"$\pi$"], fontsize=14)
        axes_srate.set_aspect(aspect=1)
        ## Show figure ////////////////////////////////////////////////////////#
        if fig_show == True:
            plt.show()
        else:
            plt.close(fig)
        #//////////////////////////////////////////////////////////////////////////////#
        return fig



if __name__=="__main__":
    from cuprates_transport.bandstructure import BandStructure
    from scipy import linalg

    params = {
    "band_name": "Nd-LSCO",
    "a": 3.75,
    "b": 3.75,
    "c": 13.2,
    "resolution": [21, 21, 3],
    "energy_scale": 160,
    "band_params":{"mu":-0.82439881, "t": 1, "tp":-0.13642799, "tpp":0.06816836, "tz":0.06512192},
    "N_time": 1000,
    "T" : 0,
    "Bamp": 45,
    "scattering_models":["isotropic", "cos2phi"],
    "scattering_params":{"gamma_0":12.595, "gamma_k": 63.823, "power": 12},
    }

    bandObject = BandStructure(**params)
    bandObject.runBandStructure(printDoping=False)
    ## Compute conductivity
    condObject = Conductivity(bandObject, **params)
    condObject.runTransport()

    rho = linalg.inv(condObject.sigma).transpose()
    rhoxx = rho[0,0]
    rhoxy = rho[0,1]
    rhozz = rho[2,2]
    print("1band-------------")
    print("rhoxx =", rhoxx*1e8, "uOhm.cm")
    print("rhozz =", rhozz*1e5, "mOhm.cm")
    print("RH =", rhoxy * 1e9 / params["Bamp"], "mm^3 / C")
