import numpy as np
from numpy import cos, sin, pi, exp, sqrt, arctan2, cosh, arccosh
from scipy.integrate import odeint
from scipy.constants import Boltzmann, hbar, elementary_charge, physical_constants
from skimage import measure
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from copy import deepcopy
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Units ////////
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 1e-12 # 1 ps in seconds

## Constant //////
hbar = hbar # m2 kg / s
e = elementary_charge # C
kB = Boltzmann # J / K
kB = kB / meV # meV / K

## This coefficient takes into accound all units and constant to prefactor the movement equation
units_move_eq =  e * Angstrom**2 * picosecond * meV / hbar**2

## This coefficient takes into accound all units and constant to prefactor Chambers formula
units_chambers = 2 * e**2 / (2*pi)**3 * meV * picosecond / Angstrom / hbar**2


class Conductivity:
    def __init__(self, bandObject, Bamp, Bphi=0, Btheta=0, N_time=500,
                 T=0, dfdE_cut_percent=0.001, N_epsilon=20,
                 gamma_0=15, a_epsilon = 0, a_abs_epsilon = 0, a_epsilon_2 = 0,
                 gamma_dos_max=0,
                 gamma_k=0, power=2, az=0,
                 factor_arcs=1,
                 gamma_step=30, phi_step=np.pi/6,
                 **trash):

        # Band object
        self.bandObject = deepcopy(bandObject)

        # Magnetic field in degrees
        self._Bamp   = Bamp # in Tesla
        self._Btheta = Btheta
        self._Bphi   = Bphi
        self._B_vector = self.BFunc() # np array fo Bx,By,Bz
        self.omegac_tau = None

        # Temperature and energy integration
        self._T = T # in Kelvin
        self._N_epsilon = N_epsilon
        self._dfdE_cut_percent = dfdE_cut_percent
        if self._T != 0:
            self.epsilon_array = np.linspace(-self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                              self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                              self._N_epsilon)


        # Scattering rate
        self.gamma_0       = gamma_0 # in THz
        self.a_epsilon     = a_epsilon # unit less
        self.a_abs_epsilon = a_abs_epsilon # unit less
        self.a_epsilon_2   = a_epsilon_2 # unit less
        self.gamma_dos_max = gamma_dos_max # in THz
        self.gamma_k       = gamma_k # in THz
        self.power         = power
        self.factor_arcs   = factor_arcs # factor * gamma_0 outsite AF FBZ
        self.gamma_step    = gamma_step
        self.phi_step      = phi_step

        # Time parameters
        self.time_max = 8 * self.tau_total_max()  # in picoseconds
        self._N_time = N_time # number of steps in time
        self.dtime = self.time_max / self.N_time
        self.time_array = np.arange(0, self.time_max, self.dtime)
        self.dtime_array = np.append(0, self.dtime * np.ones_like(self.time_array))[:-1] # integrand for tau_function

        # Time-dependent kf, vf
        self.kft = np.empty(1)
        self.vft = np.empty(1)
        self.tau = np.empty(1) # array[i0, i_t] with i0 index of the initial index
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
        self._B_vector = self.BFunc()
    Bamp = property(_get_Bamp, _set_Bamp)

    def _get_Bphi(self):
        return self._Bphi
    def _set_Bphi(self, Bphi):
        self._Bphi = Bphi
        self._B_vector = self.BFunc()
    Bphi = property(_get_Bphi, _set_Bphi)

    def _get_Btheta(self):
        return self._Btheta
    def _set_Btheta(self, Btheta):
        self._Btheta = Btheta
        self._B_vector = self.BFunc()
    Btheta = property(_get_Btheta, _set_Btheta)

    def _get_N_time(self):
        return self._N_time
    def _set_N_time(self, N_time):
        self._N_time = N_time
        self.dtime = self.time_max / self._N_time
        self.time_array = np.arange(0, self.time_max, self.dtime) # integrand for tau_function
        self.dtime_array = np.append(0, self.dtime * np.ones_like(self.time_array))[:-1]
    N_time = property(_get_N_time, _set_N_time)

    def _get_T(self):
        return self._T
    def _set_T(self, T):
        self._T = T
        if self._T != 0:
            self.epsilon_array = np.linspace(-self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                              self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                              self._N_epsilon)
    T = property(_get_T, _set_T)

    def _get_N_epsilon(self):
        return self._N_epsilon
    def _set_N_epsilon(self, N_epsilon):
        self._N_epsilon = N_epsilon
        self.epsilon_array = np.linspace(-self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                          self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                          self._N_epsilon)
    N_epsilon = property(_get_N_epsilon, _set_N_epsilon)

    def _get_dfdE_cut_percent(self):
        return self._dfdE_cut_percent
    def _set_dfdE_cut_percent(self, dfdE_cut_percent):
        self._dfdE_cut_percent  = dfdE_cut_percent
        self.epsilon_array = np.linspace(-self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                          self.energyCutOff(self._dfdE_cut_percent * np.abs(self.dfdE(0))),
                                          self._N_epsilon)
    dfdE_cut_percent = property(_get_dfdE_cut_percent, _set_dfdE_cut_percent)


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def runTransport(self):
        if self._T != 0:
            #!!!! Add a warning if bandObject not already discretized
            self.dos_k_epsilon      = {}
            self.dkf_epsilon        = {}
            self.kft_epsilon        = {}
            self.vft_epsilon        = {}
            self.t_o_tau_epsilon = {}

            for epsilon in self.epsilon_array:
                self.bandObject.runBandStructure(epsilon = epsilon, printDoping=False)
                self.solveMovementFunc()
                self.t_o_tau_func(epsilon)
                self.dos_k_epsilon[epsilon]      = self.bandObject.dos_k
                self.dkf_epsilon[epsilon]        = self.bandObject.dkf
                self.kft_epsilon[epsilon]        = self.kft
                self.vft_epsilon[epsilon]        = self.vft
                self.t_o_tau_epsilon[epsilon]    = self.t_o_tau
                ## !!!!  Do not forget to update scattering rates !!! ##
                ## Create properties for tmax, etc.
            self.bandObject.runBandStructure(epsilon = 0, printDoping=False)
            # this lasr one is to be sure the bandObject is at the FS at the end
        else:
            self.solveMovementFunc()
            self.t_o_tau_func()

        self.gamma_tot_max = 1 / self.tau_total_min() # in THz
        self.gamma_tot_min = 1 / self.tau_total_max() # in THz


    def BFunc(self):
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


    def solveMovementFunc(self):
        len_t = self.time_array.shape[0]
        len_kf = self.bandObject.kf.shape[1]

        ## Magnetic Field ON
        if self.Bamp != 0:
            # Flatten to get all the initial kf solved at the same time
            self.bandObject.kf.shape = (3 * len_kf,)
            # Sovle differential equation
            self.kft = odeint(self.diffEqFunc, self.bandObject.kf, self.time_array, rtol = 1e-4, atol = 1e-4).transpose()
            # Reshape arrays
            self.bandObject.kf.shape = (3, len_kf)
            self.kft.shape = (3, len_kf, len_t)
            # Velocity function of time
            self.vft = np.empty_like(self.kft, dtype = np.float64)
            self.vft[0, :, :], self.vft[1, :, :], self.vft[2, :, :] = self.bandObject.v_3D_func(self.kft[0, :, :], self.kft[1, :, :], self.kft[2, :, :])
        ## Magnetic Field OFF
        else:
            self.kft = np.empty((3, len_kf, 1), dtype = np.float64)
            self.vft = np.empty((3, len_kf, 1), dtype = np.float64)
            self.kft[0, :, 0], self.kft[1, :, 0], self.kft[2, :, 0] = self.bandObject.kf[0, :], self.bandObject.kf[1, :], self.bandObject.kf[2, :]
            self.vft[0, :, 0], self.vft[1, :, 0], self.vft[2, :, 0] = self.bandObject.vf[0, :], self.bandObject.vf[1, :], self.bandObject.vf[2, :]


    def diffEqFunc(self, k, t):
        len_k = int(k.shape[0]/3)
        k.shape = (3, len_k) # reshape the flatten k
        vx, vy, vz =  self.bandObject.v_3D_func(k[0,:], k[1,:], k[2,:])
        dkdt = ( - units_move_eq ) * self.crossProductVectorized(vx, vy, vz)
        dkdt.shape = (3*len_k,) # flatten k again
        return dkdt


    def factor_arcs_Func(self, kx, ky, kz):
        # line ky = kx + pi
        d1 = ky * self.bandObject.b - kx * self.bandObject.a - pi  # line ky = kx + pi
        d2 = ky * self.bandObject.b - kx * self.bandObject.a + pi  # line ky = kx - pi
        d3 = ky * self.bandObject.b + kx * self.bandObject.a - pi  # line ky = -kx + pi
        d4 = ky * self.bandObject.b + kx * self.bandObject.a + pi  # line ky = -kx - pi

        is_in_FBZ_AF = np.logical_and((d1 <= 0)*(d2 >= 0), (d3 <= 0)*(d4 >= 0))
        is_out_FBZ_AF = np.logical_not(is_in_FBZ_AF)
        factor_out_of_FBZ_AF = np.ones_like(kx)
        factor_out_of_FBZ_AF[is_out_FBZ_AF] = self.factor_arcs
        return factor_out_of_FBZ_AF


    def omegac_tau_func(self):
        dks = self.bandObject.dks / Angstrom # in m^-1
        kf = self.bandObject.kf
        vf = self.bandObject.vf * meV * Angstrom # in Joule.m (because in the code vf is not divided by hbar)
        vf_perp = sqrt(vf[0, :]**2 + vf[1, :]**2)  # vf perp to B, in Joule.m
        prefactor = (hbar)**2 / (2 * pi * e * self._Bamp) / self.bandObject.numberOfKz # divide by the number of kz to average over all kz
        inverse_omegac_tau = \
            prefactor * np.sum(dks / vf_perp / (picosecond * self.tau_total_func(kf[0, :], kf[1, :], kf[2, :],
                                                                                 vf[0, :], vf[1, :], vf[2, :])))
        self.omegac_tau = 1 / inverse_omegac_tau


    def gamma_DOS_Func(self, vx, vy, vz):
        dos = 1 / sqrt( vx**2 + vy**2 + vz**2 )
        dos_max = np.max(self.bandObject.dos_k)  # value to normalize the DOS to a quantity without units
        return self.gamma_dos_max * (dos / dos_max)


    def gamma_k_Func(self, kx, ky, kz):
        ## Make sure kx and ky are in the FBZ to compute Phi.
        kx = np.remainder(kx + pi / self.bandObject.a, 2*pi / self.bandObject.a) - pi / self.bandObject.a
        ky = np.remainder(ky + pi / self.bandObject.b, 2*pi / self.bandObject.b) - pi / self.bandObject.b
        phi = arctan2(ky, kx)
        gamma_k = self.gamma_k * np.abs(cos(2*phi))**self.power # / (1 + self.az*abs(sin(kz*self.bandObject.c/2)))

        quad_phi = phi % (np.pi/2)
        mask = (quad_phi < self.phi_step) | (quad_phi > (np.pi/2 - self.phi_step))
        gamma_k += self.gamma_step * mask

        return gamma_k

    def tau_total_func(self, kx, ky, kz, vx, vy, vz, epsilon = 0):
        """Computes the total lifetime based on the input model
        for the scattering rate"""

        epsilon = epsilon / self.bandObject.t

        ## Gamma epsilon^coeff_k
        # gammaTot *= self.gamma_0 * np.ones_like(kx)
        # if self.gamma_k!=0:
        #     # phi = arctan2(ky, kx)
        #     gammaTot += self.gamma_k_Func(kx, ky, kz) * (self.a_epsilon * epsilon + self.a_abs_epsilon * np.abs(epsilon))

        # ## Gamma |epsilon|*cos(2*phi)
        # gammaTot = (1 + self.a_epsilon_2 * epsilon**2)
        # gammaTot *= self.gamma_0 * np.ones_like(kx)
        # if self.gamma_k!=0:
        #     gammaTot += self.gamma_k_Func(kx, ky, kz) * (self.a_epsilon * epsilon + self.a_abs_epsilon * np.abs(epsilon))

        # gammaTot = 1 + self.a_epsilon * epsilon + self.a_abs_epsilon * sqrt((kB*self.T)**2 + np.abs(epsilon)**2) + self.a_epsilon_2*((kB*self.T)**2 + epsilon**2)
        gammaTot = 1 + self.a_epsilon * epsilon + self.a_abs_epsilon * np.abs(epsilon) + self.a_epsilon_2 * epsilon**2
        gammaTot *= self.gamma_0 * np.ones_like(kx)
        if self.gamma_k!=0:
            gammaTot += self.gamma_k_Func(kx, ky, kz)
        if self.gamma_dos_max!=0:
            gammaTot += self.gamma_DOS_Func(vx, vy, vz)
        if self.factor_arcs!=1:
            gammaTot *= self.factor_arcs_Func(kx, ky, kz)

        return 1/gammaTot


    def t_o_tau_func(self, espilon = 0):
        ## Integral from 0 to t of dt' / tau( k(t') ) or dt' * gamma( k(t') )
        ## Magnetic Field ON
        if self.Bamp !=0:
            self.t_o_tau = np.cumsum( self.dtime_array /
                                     (self.tau_total_func(self.kft[0, :, :],
                                                          self.kft[1, :, :],
                                                          self.kft[2, :, :],
                                                          self.vft[0, :, :],
                                                          self.vft[1, :, :],
                                                          self.vft[2, :, :],
                                                          espilon)), axis = 1)
        ## Magnetic Field OFF
        else:
            self.t_o_tau = 1 / self.tau_total_func(self.kft[0, :, 0],
                                                   self.kft[1, :, 0],
                                                   self.kft[2, :, 0],
                                                   self.vft[0, :, 0],
                                                   self.vft[1, :, 0],
                                                   self.vft[2, :, 0],
                                                   espilon)

    def tau_total_max(self):
        # Compute the tau_max (the longest time between two collisions)
        # to better integrate from 0 --> 8 * 1 / gamma_min (instead of infinity)
        kf = self.bandObject.kf
        vf = self.bandObject.vf
        return np.max(self.tau_total_func(kf[0, :], kf[1, :], kf[2, :],
                                          vf[0, :], vf[1, :], vf[2, :]))

    def tau_total_min(self):
        kf = self.bandObject.kf
        vf = self.bandObject.vf
        return np.min(self.tau_total_func(kf[0, :], kf[1, :], kf[2, :],
                                          vf[0, :], vf[1, :], vf[2, :]))


    def velocity_product(self, kft, vft, t_o_tau, i, j):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0: vif = vxf """

        if self.Bamp != 0:
            self.v_product = vft[i, :, 0] * np.sum(vft[j, :, :] * exp(-t_o_tau) * self.dtime, axis=1)
        else:
            self.v_product = vft[i, :, 0] * vft[j, :, 0] * (1 / t_o_tau)
        return self.v_product

    def sigma_epsilon(self, dos_k, dkf, kft, vft, t_o_tau, i, j):
        sigma_epsilon = units_chambers / self.bandObject.numberOfBZ * \
                        np.sum( dkf * dos_k * self.velocity_product(kft, vft, t_o_tau, i=i, j=j) )
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

    def chambersFunc(self, i, j, coeff_name="sigma"):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0 and j = 1 : sigma[i,j] = sigma_xy """

        #!!! Add a error message if asking for alpha and beta at T != 0

        if self._T == 0:
            coeff_tot = self.sigma_epsilon(self.bandObject.dos_k,
                                                  self.bandObject.dkf,
                                                  self.kft, self.vft,
                                                  self.t_o_tau,
                                                  i=i, j=j)
            self.sigma[i, j] = coeff_tot
        else:
            coeff_tot = 0
            d_epsilon = self.epsilon_array[1] - self.epsilon_array[0]
            for epsilon in self.epsilon_array:
                sigma_epsilon = self.sigma_epsilon(self.dos_k_epsilon[epsilon],
                                                   self.dkf_epsilon[epsilon],
                                                   self.kft_epsilon[epsilon],
                                                   self.vft_epsilon[epsilon],
                                                   self.t_o_tau_epsilon[epsilon],
                                                   i=i, j=j)
                # Sum over the energie
                coeff_tot += d_epsilon * (- self.dfdE(epsilon)) * \
                             self.integrand_coeff(epsilon, coeff_name) * sigma_epsilon

            # Send to right transport coefficient
            if coeff_name == "sigma":
                self.sigma[i, j] = coeff_tot
            elif coeff_name == "alpha":
                self.alpha[i, j] = coeff_tot
            elif coeff_name == "beta":
                self.beta[i, j]  = coeff_tot
            else:
                print("!Warming! You have not enter a correct coefficient name")

        return coeff_tot

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



    ## Figures ////////////////////////////////////////////////////////////////#

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

    def figScatteringColor(self, kz=0, mesh_xy=501):
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 5.6))
        fig.subplots_adjust(left=0.10, right=0.85, bottom=0.20, top=0.9)

        kx_a = np.linspace(-pi / self.bandObject.a, pi / self.bandObject.a, mesh_xy)
        ky_a = np.linspace(-pi / self.bandObject.b, pi / self.bandObject.b, mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = self.bandObject.e_3D_func(kxx, kyy, kz)
        contours = measure.find_contours(bands, 0)

        gamma_max_list = []
        gamma_min_list = []
        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / self.bandObject.a
            ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / self.bandObject.b
            vx, vy, vz = self.bandObject.v_3D_func(kx, ky, kz)

            gamma_kz = 1 / self.tau_total_func(kx, ky, kz, vx, vy, vz)
            gamma_max_list.append(np.max(gamma_kz))
            gamma_min_list.append(np.min(gamma_kz))

            points = np.array([kx*self.bandObject.a, ky*self.bandObject.b]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=plt.get_cmap('gnuplot'))
            lc.set_array(gamma_kz)
            lc.set_linewidth(4)

            line = axes.add_collection(lc)

        gamma_max = max(gamma_max_list)
        gamma_min = min(gamma_min_list)
        line.set_clim(gamma_min, gamma_max)
        cbar = fig.colorbar(line, ax=axes)
        cbar.minorticks_on()
        cbar.ax.set_ylabel(r'$\Gamma_{\rm tot}$ ( THz )', rotation=270, labelpad=40)

        kz = np.round(kz/(np.pi/self.bandObject.c), 1)
        fig.text(0.97, 0.05, r"$k_{\rm z}$ = " + "{0:g}".format(kz) + r"$\pi$/c", ha="right", color="r")
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
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 4.6))
        fig.subplots_adjust(left=0.20, right=0.8, bottom=0.20, top=0.9)
        axes2 = axes.twinx()
        axes2.set_axisbelow(True)

        ###
        kx_a = np.linspace(-pi / self.bandObject.a, pi / self.bandObject.a, mesh_xy)
        ky_a = np.linspace(-pi / self.bandObject.b, pi / self.bandObject.b, mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = self.bandObject.e_3D_func(kxx, kyy, kz)
        contours = measure.find_contours(bands, 0)

        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / self.bandObject.a
            ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / self.bandObject.b
            vx, vy, vz = self.bandObject.v_3D_func(kx, ky, kz)

            gamma_kz = 1 / self.tau_total_func(kx, ky, kz, vx, vy, vz)

            phi = np.rad2deg(np.arctan2(ky,kx))

            line = axes2.plot(phi, vz)
            plt.setp(line, ls ="", c = '#80ff80', lw = 3, marker = "o", mfc = '#80ff80', ms = 3, mec = "#7E2320", mew= 0)  # set properties

            line = axes.plot(phi, gamma_kz)
            plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 3, mec = "#7E2320", mew= 0)  # set properties

        axes.set_xlim(0, 180)
        axes.set_xticks([0, 45, 90, 135, 180])
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$\phi$", labelpad = 8)
        axes.set_ylabel(r"$\Gamma_{\rm tot}$ ( THz )", labelpad=8)
        axes2.set_ylabel(r"$v_{\rm z}$", rotation = 270, labelpad =25, color="#80ff80")



        kz = np.round(kz/(np.pi/self.bandObject.c), 1)
        fig.text(0.97, 0.05, r"$k_{\rm z}$ = " + "{0:g}".format(kz) + r"$\pi$/c", ha="right", color="r", fontsize = 20)
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
        kx = np.linspace(-pi / self.bandObject.a, pi / self.bandObject.a, mesh_graph)
        ky = np.linspace(-pi / self.bandObject.b, pi / self.bandObject.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

        fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6))
        fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91)

        fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

        line = axes.contour(kxx*self.bandObject.a, kyy*self.bandObject.b, self.bandObject.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
        line = axes.plot(self.kft[0, index_kf,:]*self.bandObject.a, self.kft[1, index_kf,:]*self.bandObject.b)
        plt.setp(line, ls ="-", c = 'b', lw = 1, marker = "", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0) # trajectory
        line = axes.plot(self.bandObject.kf[0, index_kf]*self.bandObject.a, self.bandObject.kf[1, index_kf]*self.bandObject.b)
        plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 4.5, mec = "b", mew= 1.5)  # starting point
        line = axes.plot(self.kft[0, index_kf, -1]*self.bandObject.a, self.kft[1, index_kf, -1]*self.bandObject.b)
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

        line = axes.plot(epsilon_cut, dfdE_cut)
        plt.setp(line, ls ="", c = 'k', lw = 3, marker = "s", mfc = 'k', ms = 6, mec = "#7E2320", mew= 0)  # set properties
        line = axes.plot(-epsilon_cut, dfdE_cut)
        plt.setp(line, ls ="", c = 'k', lw = 3, marker = "s", mfc = 'k', ms = 6, mec = "#7E2320", mew= 0)  # set properties


        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$\epsilon$", labelpad = 8)
        axes.set_ylabel(r"-$df/d\epsilon$ ( units of t )", labelpad = 8)

        axes.locator_params(axis = 'y', nbins = 6)

        plt.show()

    #---------------------------------------------------------------------------

    def figParameters(self, fig_show=True):

        # (1,1) means one plot, and figsize is w x h in inch of figure
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
        # adjust the box of axes regarding the figure size
        fig.subplots_adjust(left=0.15, right=0.25, bottom=0.18, top=0.95)
        axes.remove()

        # Band name
        fig.text(0.45, 0.92, "Band :: " +
                    self.bandObject.bandname, fontsize=20, color='#00d900')
        try:
            self.bandObject.M
            fig.text(0.41, 0.92, "AF", fontsize=20,
                        color="#FF0000")
        except:
            None

        # Band Formulas
        fig.text(0.45, 0.445, "Band formula", fontsize=16,
                    color='#008080')
        fig.text(0.45, 0.4, r"$a$ = " + "{0:.2f}".format(self.bandObject.a) + r" $\AA$  ::  " +
                            r"$b$ = " + "{0:.2f}".format(self.bandObject.b) + r" $\AA$  ::  " +
                            r"$c$ = " + "{0:.2f}".format(self.bandObject.c) + r" $\AA$", fontsize=10)

        # r"$c$ " + "{0:.2f}".format(self.bandObject.c)
        bandFormulaE2D = r"$\epsilon_{\rm k}^{\rm 2D}$ = - $\mu$" +\
            r" - 2$t$ (cos($k_{\rm x}a$) + cos($k_{\rm y}b$))" +\
            r" - 4$t^{'}$ (cos($k_{\rm x}a$) cos($k_{\rm y}b$))" + "\n" +\
            r"          - 2$t^{''}$ (cos(2$k_{\rm x}a$) + cos(2$k_{\rm y}b$))" + "\n"
        fig.text(0.45, 0.27, bandFormulaE2D, fontsize=10)

        bandFormulaEz = r"$\epsilon_{\rm k}^{\rm z}$   =" +\
            r" - 2$t_{\rm z}$ cos($k_{\rm z}c/2$) cos($k_{\rm x}a/2$) cos($k_{\rm y}b/2$) (cos($k_{\rm x}a$) - cos($k_{\rm y}b$))$^2$" + "\n" +\
            r"          - 2$t_{\rm z}^{'}$ cos($k_{\rm z}c/2$)"
        fig.text(0.45, 0.21, bandFormulaEz, fontsize=10)


        # AF Band Formula
        try:
            self.bandObject.M
            if self.bandObject.electronPocket == True:
                sign_symbol = "+"
            else:
                sign_symbol = "-"
            AFBandFormula = r"$\epsilon_{\rm k}^{\rm 3D " + sign_symbol + r"}$ = 1/2 ($\epsilon_{\rm k}^{\rm 2D}$ + $\epsilon_{\rm k+Q}^{\rm 2D}$) " +\
                sign_symbol + \
                r" $\sqrt{1/4(\epsilon_{\rm k}^{\rm 2D} - \epsilon_{\rm k+Q}^{\rm 2D})^2 + \Delta_{\rm AF}^2}$ + $\epsilon_{\rm k}^{\rm z}$"
            fig.text(0.45, 0.15, AFBandFormula,
                        fontsize=10, color="#FF0000")
        except:
            bandFormulaE3D = r"$\epsilon_{\rm k}^{\rm 3D}$   = $\epsilon_{\rm k}^{\rm 2D}$ + $\epsilon_{\rm k}^{\rm z}$"
            fig.text(0.45, 0.15, bandFormulaE3D, fontsize=10)


        # Scattering Formula
        fig.text(0.45, 0.08, "Scattering formula",
                    fontsize=16, color='#008080')
        scatteringFormula = r"$\Gamma_{\rm tot}$ = [ $\Gamma_{\rm 0}$ + " + \
            r"$\Gamma_{\rm k}$ |cos$^{\rm n}$(2$\phi$)| + $\Gamma_{\rm DOS}^{\rm max}$ (DOS / DOS$^{\rm max}$) ] $A_{\rm arcs}$"
        fig.text(0.45, 0.03, scatteringFormula, fontsize=10)

        # Parameters Bandstructure
        fig.text(0.45, 0.85, "Band Parameters", fontsize=16,
                    color='#008080')
        label_parameters = [r"$t$     =  " + "{0:.1f}".format(self.bandObject.t) + "    meV",
                            r"$\mu$    =  " +
                            "{0:+.3f}".format(self.bandObject.mu) +
                            r"   $t$",
                            r"$t^\prime$    =  " +
                            "{0:+.3f}".format(self.bandObject.tp) +
                            r"   $t$",
                            r"$t^{\prime\prime}$   =  " + "{0:+.3f}".format(
                                self.bandObject.tpp) + r"   $t$",
                            r"$t_{\rm z}$    =  " + "{0:+.3f}".format(
                                self.bandObject.tz) + r"   $t$",
                            r"$t_{\rm z}^{\prime}$    =  " +
                            "{0:+.3f}".format(self.bandObject.tz2) + r"   $t$",
                            r"$t_{\rm z}^{\prime\prime}$    =  " +
                            "{0:+.3f}".format(self.bandObject.tz3) + r"   $t$",
                            r"$t_{\rm z}^{\prime\prime\prime}$    =  " +
                            "{0:+.3f}".format(self.bandObject.tz4) + r"   $t$"
                            ]
        try:  # if it is a AF band
            self.bandObject.M
            label_parameters.append(
                r"$\Delta_{\rm AF}$ =  " + "{0:+.3f}".format(self.bandObject.M) + r"   $t$")
        except:
            None

        h_label = 0.80
        for label in label_parameters:
            fig.text(0.45, h_label, label, fontsize=14)
            h_label -= 0.043

        # Band filling
        fig.text(0.72, 0.85, "Band Filling =", fontsize=16,
                    color='#008080')
        fig.text(0.855, 0.85, "{0:.3f}".format(
                    self.bandObject.n), fontsize=16, color='#000000')

        # Scattering parameters
        fig.text(0.72, 0.79, "Scattering Parameters",
                    fontsize=16, color='#008080')
        label_parameters = [
            r"$\Gamma_{\rm 0}$       = " + "{0:.1f}".format(self.gamma_0) +
            "   THz",
            r"$\Gamma_{\rm DOS}^{\rm max}$   = " +
            "{0:.1f}".format(self.gamma_dos_max) + "   THz",
            r"$\Gamma_{\rm k}$       = " + "{0:.1f}".format(self.gamma_k) +
            "   THz",
            r"$n$         = " + "{0:.1f}".format(self.power),
            r"$A_{\rm arcs}$   = " + "{0:.1f}".format(self.factor_arcs),
            r"$\Gamma_{\rm tot}^{\rm max}$    = " +
            "{0:.1f}".format(self.gamma_tot_max) + "   THz",
            r"$\Gamma_{\rm tot}^{\rm min}$     = " +
            "{0:.1f}".format(self.gamma_tot_min) + "   THz",
        ]
        h_label = 0.74
        for label in label_parameters:
            fig.text(0.72, h_label, label, fontsize=14)
            h_label -= 0.043

        ## Inset FS ///////////////////////////////////////////////////////////#
        a = self.bandObject.a
        b = self.bandObject.b
        c = self.bandObject.c

        mesh_graph = 201
        kx = np.linspace(-pi/a, pi/a, mesh_graph)
        ky = np.linspace(-pi/b, pi/b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing='ij')

        axes_FS = plt.axes([-0.02, 0.56, .4, .4])
        axes_FS.set_aspect(aspect=1)
        axes_FS.contour(kxx, kyy, self.bandObject.e_3D_func(
            kxx, kyy, 0), 0, colors='#FF0000', linewidths=1)
        axes_FS.contour(kxx, kyy, self.bandObject.e_3D_func(
            kxx, kyy, pi / c), 0, colors='#00DC39', linewidths=1)
        axes_FS.contour(kxx, kyy, self.bandObject.e_3D_func(
            kxx, kyy, 2 * pi / c), 0, colors='#6577FF', linewidths=1)
        fig.text(0.30, 0.67, r"$k_{\rm z}$", fontsize=14)
        fig.text(0.30, 0.63, r"0", color='#FF0000', fontsize=14)
        fig.text(0.30, 0.60, r"$\pi$/c", color='#00DC39', fontsize=14)
        fig.text(0.30, 0.57, r"$2\pi$/c", color='#6577FF', fontsize=14)

        axes_FS.set_xlabel(r"$k_{\rm x}$", labelpad=0, fontsize=14)
        axes_FS.set_ylabel(r"$k_{\rm y}$", labelpad=-5, fontsize=14)

        axes_FS.set_xticks([-pi/a, 0., pi/a])
        axes_FS.set_xticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=14)
        axes_FS.set_yticks([-pi/b, 0., pi/b])
        axes_FS.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=14)
        # axes_FS.tick_params(axis='x', which='major', pad=7)
        # axes_FS.tick_params(axis='y', which='major', pad=8)

        ## Inset Scattering rate ////////////////////////////////////////////////#
        axes_srate = plt.axes([-0.02, 0.04, .4, .4])
        axes_srate.set_aspect(aspect=1)

        mesh_xy = 501
        kx_a = np.linspace(-pi / self.bandObject.a, pi / self.bandObject.a,
                           mesh_xy)
        ky_a = np.linspace(-pi / self.bandObject.b, pi / self.bandObject.b,
                           mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = self.bandObject.e_3D_func(kxx, kyy, 0)
        contours = measure.find_contours(bands, 0)

        gamma_max_list = []
        gamma_min_list = []
        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] /
                  (mesh_xy - 1) - 0.5) * 2 * pi / self.bandObject.a
            ky = (contour[:, 1] /
                  (mesh_xy - 1) - 0.5) * 2 * pi / self.bandObject.b
            vx, vy, vz = self.bandObject.v_3D_func(kx, ky, 0)

            gamma_kz0 = 1 / self.tau_total_func(kx, ky, 0, vx, vy, vz)
            gamma_max_list.append(np.max(gamma_kz0))
            gamma_min_list.append(np.min(gamma_kz0))

            points = np.array([kx * self.bandObject.a,
                               ky * self.bandObject.b]).T.reshape(-1, 1, 2)
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

        fig.text(0.295, 0.405, r"$k_{\rm z}$=0", fontsize=10, ha="right")
        axes_srate.tick_params(axis='x', which='major')
        axes_srate.tick_params(axis='y', which='major')
        axes_srate.set_xlabel(r"$k_{\rm x}$", fontsize=14)
        axes_srate.set_ylabel(r"$k_{\rm y}$", fontsize=14, labelpad=-5)

        axes_srate.set_xticks([-pi, 0., pi])
        axes_srate.set_xticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=14)
        axes_srate.set_yticks([-pi, 0., pi])
        axes_srate.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=14)

        ## Show figure ////////////////////////////////////////////////////////#
        if fig_show == True:
            plt.show()
        else:
            plt.close(fig)
        #//////////////////////////////////////////////////////////////////////////////#

        return fig
