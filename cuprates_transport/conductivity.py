import numpy as np
from numpy import cos, sin, pi, exp, sqrt, arctan2, cosh, arccosh
from scipy.integrate import odeint
from scipy.constants import Boltzmann, hbar, elementary_charge, physical_constants
from skimage import measure
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from copy import deepcopy
from .conductivity_gammas import *
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
    from .conductivity_plotting import figParameters, figCumulativevft, figOnekft, \
        figScatteringPhi, figScatteringColor
    def __init__(self, bandObject, Bamp, Bphi=0, Btheta=0, N_time=500,
                 T=0, dfdE_cut_percent=0.001, N_epsilon=20,
                 gamma_0=15, a_epsilon = 0, a_abs_epsilon = 0, a_epsilon_2 = 0, a_T = 0, a_T2 = 0,
                 a_asym=0, p_asym=0,
                 gamma_dos_max=0,
                 gamma_cmfp_max=0,
                 gamma_k=0, power=2,
                 a0=0, a1=0, a2=0, a3=0, a4=0, a5=0,
                 gamma_kpi4=0, powerpi4=0,
                 l_path=0,
                 gamma_step=0, phi_step=0,
                 factor_arcs=1,
                 **trash):

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


        # Scattering rate energy-dependent
        self.a_epsilon     = a_epsilon # in THz/meV
        self.a_abs_epsilon = a_abs_epsilon # in THz/meV
        self.a_epsilon_2   = a_epsilon_2 # in THz/meV^2
        self.a_T           = a_T # unitless
        self.a_T2          = a_T2 # unitless

        # Scattering rate asymmetric Antoine Georger
        self.a_asym = a_asym # in THz/meV
        self.p_asym = p_asym # unitless

        # Scattering rate k-dependent
        self.gamma_0       = gamma_0 # in THz
        self.gamma_dos_max = gamma_dos_max # in THz
        self.gamma_cmfp_max = gamma_cmfp_max
        self.gamma_k       = gamma_k # in THz
        self.power         = power
        self.gamma_step    = gamma_step
        self.phi_step      = np.deg2rad(phi_step)
        self.factor_arcs   = factor_arcs # factor * gamma_0 outsite AF FBZ
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.gamma_kpi4 = gamma_kpi4
        self.powerpi4 = powerpi4
        self.l_path = l_path # in Angstrom, mean free path for gamma_vF

        # Time parameters
        self.time_max = 8 * self.tau_total_max()  # in picoseconds
        self._N_time = N_time # number of steps in time
        self.dtime = self.time_max / self.N_time
        self.time_array = np.arange(0, self.time_max, self.dtime)
        self.dtime_array = np.append(0, self.dtime * np.ones_like(self.time_array))[:-1] # integrand for tau_function

        ## Precision differential equation solver
        self.rtol = 1e-4 # default is 1.49012e-8
        self.atol = 1e-4 # default is 1.49012e-8

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


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def runTransport(self):
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
            vft0, vft1, vft2 = self.bandObject.v_3D_func(kft0, kft1, kft2)
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
        vx, vy, vz =  self.bandObject.v_3D_func(k[0,:], k[1,:], k[2,:])
        dkdt = ( - units_move_eq ) * self.crossProductVectorized(vx, vy, vz)
        dkdt.shape = (3*len_k,) # flatten k again
        return dkdt


    def omegac_tau_func(self):
        dks = self.bandObject.dks / Angstrom # in m^-1
        kf = self.bandObject.kf
        vf = self.bandObject.vf * meV * Angstrom # in Joule.m (because in the code vf is not divided by hbar)
        kf_perp = sqrt(kf[0, :]**2 + kf[1, :]**2) / Angstrom  # kf perp to B in m
        vf_perp = sqrt(vf[0, :]**2 + vf[1, :]**2)  # vf perp to B, in Joule.m
        prefactor = (hbar)**2 / (2 * pi * e * self._Bamp)

        ## Function of k
        kf0, kf1, kf2 = kf[0, :], kf[1, :], kf[2, :]
        vf0, vf1, vf2 = vf[0, :], vf[1, :], kf[2, :]
        inverse_omegac_tau_k = (prefactor * 2*pi*kf_perp / vf_perp / picosecond *
                                self.tau_total_func(kf0, kf1, kf2, vf0, vf1, vf2))
        self.omegac_tau_k = 1 / inverse_omegac_tau_k

        # ## Integrated over the Fermi surface
        # inverse_omegac_tau = np.sum(prefactor * dks / vf_perp / (picosecond * self.tau_total_func(kf[0, :], kf[1, :], kf[2, :], vf[0, :], vf[1, :], vf[2, :]))) / self.bandObject.res_z  # divide by the number of kz to average over all kz
        # self.omegac_tau = 1 / inverse_omegac_tau


    ## List of scatterint rate models --------------------------------------------
    def factor_arcs_func(self, kx, ky, kz):
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

    def tau_total_func(self, kx, ky, kz, vx, vy, vz, epsilon = 0):
        """Computes the total lifetime based on the input model
        for the scattering rate"""

        ## Initialize
        gamma_tot = self.gamma_0 * np.ones_like(kx)

        if self.a_epsilon!=0 or self.a_abs_epsilon!=0 or self.a_T!=0:
            gamma_tot += gamma_skew_marginal_fl(self, epsilon)
        if self.a_epsilon_2!=0 or self.a_T2!=0:
            gamma_tot += gamma_fl(self, epsilon)
        if self.a_asym!=0 or self.p_asym!=0:
            gamma_tot += gamma_skew_planckian(self, epsilon)
        if self.a0!=0 or self.a1!=0 or self.a2!=0 or self.a3!=0 or self.a4!=0 or self.a5!=0:
            # gamma_tot += gamma_cosk_coskpi4_func(self, kx, ky, kz)
            ##gamma_tot += gamma_poly_func(self, kx, ky, kz)
            gamma_tot += gamma_sinn_cosm(self, kx, ky, kz)
            # gamma_tot += gamma_tanh_func(self, x, ky, kz)
            # gamma_tot += gamma_ndlsco_tl2201_func(self, kx, ky, kz)
        if self.gamma_kpi4!=0:
            gamma_tot += gamma_coskpi4_func(self, kx, ky, kz)
        if self.gamma_k!=0:
            gamma_tot += gamma_k_func(self, kx, ky, kz)
        if self.gamma_step!=0:
            gamma_tot += gamma_step_func(self, kx, ky, kz)
        if self.gamma_dos_max!=0:
            gamma_tot += gamma_DOS_func(self, vx, vy, vz)
        if self.gamma_cmfp_max!=0:
            gamma_tot += gamma_cmfp_func(self, vx, vy, vz)
        if self.l_path != 0:
            gamma_tot += gamma_vF_func(self, vx, vy, vz)
        if self.factor_arcs!=1:
            gamma_tot *= self.factor_arcs_func(kx, ky, kz)
        # return 1/(gamma_tot*(1-0.065*cos(self.bandObject.c*kz)))
        return 1/gamma_tot


    def t_o_tau_func(self, epsilon = 0):
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
                                                          epsilon)), axis = 1)
        ## Magnetic Field OFF
        else:
            self.t_o_tau = 1 / self.tau_total_func(self.kft[0, :, 0],
                                                   self.kft[1, :, 0],
                                                   self.kft[2, :, 0],
                                                   self.vft[0, :, 0],
                                                   self.vft[1, :, 0],
                                                   self.vft[2, :, 0],
                                                   epsilon)

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
        sigma_epsilon = (units_chambers / self.bandObject.numberOfBZ *
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
