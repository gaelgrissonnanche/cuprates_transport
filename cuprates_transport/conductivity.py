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
e = elementary_charge # C
kB = Boltzmann # J / K
kB = kB / meV # meV / K

## This coefficient takes into accound all units and constant to prefactor Chambers formula
units_chambers = 2 * e**2 / (2*pi)**3 * picosecond / Angstrom**2

class Conductivity:
    def __init__(self, bandObject, Bamp, Bphi=0, Btheta=0, omega=0, N_time=500,
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
        self.omega = omega # in THz
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


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
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
        dkdt = ( - e * picosecond * Angstrom / hbar ) * self.crossProductVectorized(vx, vy, vz)
        dkdt.shape = (3*len_k,) # flatten k again
        return dkdt


    ## Chambers formula's parts -------------------------------------------------#

    def velocity_product(self, kft, vft, t_o_tau):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0: vif = vxf """
        if self.Bamp != 0:
            vi = vft[:, :, 0] # (3, len_kf)
            if self.omega == 0:
                vj = np.sum(vft[:, :, :] * exp(-t_o_tau) * self.dtime, axis=2) # (3, len_kf)
            else:
                vj = np.sum(vft[:, :, :] * exp(-t_o_tau) * self.dtime
                            * exp(-1j * self.omega * self.time_array), axis=2) # (3, len_kf)
            self.v_product = np.einsum('ji,ki->jki',vi,vj) # (3, 3, len_kf)
                                        # with 3, 3 the indices for i, j in sigma
        else:
            if self.omega != 0:
                vj = np.sum(vft[:, :, :] * exp((-1j * self.omega - t_o_tau[:, None])
                    * self.time_array) * self.dtime, axis=2) # (3, len_kf)
                self.v_product = np.einsum('ji,ki->jki', vft[:, :, 0], vj)
            else:
                self.v_product = np.einsum(
                    'ji,ki->jki', vft[:, :, 0], vft[:, :, 0] * (1 / t_o_tau))
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
                                self.tau_total_func(kf0, kf1, kf2, vf0, vf1, vf2))
        self.omegac_tau_k = 1 / inverse_omegac_tau_k

        # ## Integrated over the Fermi surface
        # inverse_omegac_tau = np.sum(prefactor * dks / vf_perp / (picosecond * self.tau_total_func(kf[0, :], kf[1, :], kf[2, :], vf[0, :], vf[1, :], vf[2, :]))) / self.bandObject.res_z  # divide by the number of kz to average over all kz
        # self.omegac_tau = 1 / inverse_omegac_tau


    ## Scatterint rate models ---------------------------------------------------#

    def gamma_DOS_func(self, vx, vy, vz):
        dos = 1 / sqrt( vx**2 + vy**2 + vz**2 )
        dos_max = np.max(self.bandObject.dos_k)
        # value to normalize the DOS to a quantity without units
        return self.gamma_dos_max * (dos / dos_max)

    def gamma_cmfp_func(self, vx, vy, vz):
        vf = sqrt( vx**2 + vy**2 + vz**2 )
        vf_max = np.max(self.bandObject.vf)
        # value to normalize the DOS to a quantity without units
        return self.gamma_cmfp_max * (vf / vf_max)

    def gamma_vF_func(self, vx, vy, vz):
        """vx, vy, vz are in Angstrom.meV
        l_path is in Angstrom
        """
        vF = sqrt(vx**2 + vy**2 + vz**2) / Angstrom * 1e-12 # in Angstrom / ps
        return vF / self.l_path

    def gamma_k_func(self, kx, ky, kz):
        ## Make sure kx and ky are in the FBZ to compute Phi.
        a, b = self.bandObject.a, self.bandObject.b
        kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
        ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
        phi = arctan2(ky, kx) #+ np.pi/4
        return self.gamma_k * np.abs(cos(2*phi))**self.power

    def gamma_coskpi4_func(self, kx, ky, kz):
        ## Make sure kx and ky are in the FBZ to compute Phi.
        a, b = self.bandObject.a, self.bandObject.b
        kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
        ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
        phi = arctan2(ky, kx) #+ np.pi/4
        return self.gamma_kpi4 * np.abs(cos(2*(phi+1*pi/4)))**self.powerpi4

    def gamma_poly_func(self, kx, ky, kz):
        ## Make sure kx and ky are in the FBZ to compute Phi.
        a, b = self.bandObject.a, self.bandObject.b
        kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
        ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
        phi = arctan2(ky, kx)
        phi_p = np.abs((np.mod(phi, pi/2)-pi/4))
        return (self.a0 + np.abs(self.a1 * phi_p + self.a2 * phi_p**2 +
                self.a3 * phi_p**3 + self.a4 * phi_p**4 + self.a5 * phi_p**5))

    def gamma_sinn_cosm(self, kx, ky, kz):
        ## Make sure kx and ky are in the FBZ to compute Phi.
        a, b = self.bandObject.a, self.bandObject.b
        kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
        ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
        phi = arctan2(ky, kx) #+ np.pi/4
        return self.a0 + self.a1 * np.abs(cos(2*phi))**self.a2 + self.a3 * np.abs(sin(2*phi))**self.a4

    def gamma_tanh_func(self, kx, ky, kz):
        ## Make sure kx and ky are in the FBZ to compute Phi.
        a, b = self.bandObject.a, self.bandObject.b
        kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
        ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
        phi = arctan2(ky, kx)
        return self.a0 / np.abs(np.tanh(self.a1 + self.a2 * np.abs(cos(2*(phi+pi/4)))**self.a3))

    def gamma_step_func(self, kx, ky, kz):
        ## Make sure kx and ky are in the FBZ to compute Phi.
        a, b = self.bandObject.a, self.bandObject.b
        kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
        ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
        phi = arctan2(ky, kx)
        index_low = ((np.mod(phi, pi/2) >= (pi/4 - self.phi_step)) *
                    (np.mod(phi, pi/2) <= (pi/4 + self.phi_step)))
        index_high = np.logical_not(index_low)
        gamma_step_array = np.zeros_like(phi)
        gamma_step_array[index_high] = self.gamma_step
        return gamma_step_array

    def gamma_ndlsco_tl2201_func(self, kx, ky, kz):
        ## Make sure kx and ky are in the FBZ to compute Phi.
        a, b = self.bandObject.a, self.bandObject.b
        kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
        ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
        phi = arctan2(ky, kx)
        return (self.a0 + self.a1 * np.abs(cos(2*phi))**12 +
                self.a2 * np.abs(cos(2*phi))**2)

    def gamma_skew_marginal_fl(self, epsilon):
        return np.sqrt((self.a_epsilon * epsilon +
                        self.a_abs_epsilon * np.abs(epsilon))**2 +
                        (self.a_T * kB * meV / hbar * 1e-12 * self.T)**2)

    def gamma_fl(self, epsilon):
        return (self.a_epsilon_2 * epsilon**2 +
                self.a_T2 * (kB * meV / hbar * 1e-12 * self.T)**2)

    def gamma_skew_planckian(self, epsilon):
        x = epsilon / (kB * self.T)
        x = np.where(x == 0, 1.0e-20, x)
        return ((self.a_asym * kB * self.T) * ((x + self.p_asym)/2) * np.cosh(x/2) /
                np.sinh((x + self.p_asym)/2) / np.cosh(self.p_asym/2))

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


    # Calculates the total scattering rate using Matthiessen's rule -------------#

    def tau_total_func(self, kx, ky, kz, vx, vy, vz, epsilon = 0):
        """Computes the total lifetime based on the input model
        for the scattering rate"""

        ## Initialize
        gamma_tot = self.gamma_0 * np.ones_like(kx)

        if self.a_epsilon!=0 or self.a_abs_epsilon!=0 or self.a_T!=0:
            gamma_tot += self.gamma_skew_marginal_fl(epsilon)
        if self.a_epsilon_2!=0 or self.a_T2!=0:
            gamma_tot += self.gamma_fl(epsilon)
        if self.a_asym!=0 or self.p_asym!=0:
            gamma_tot += self.gamma_skew_planckian(epsilon)
        if self.a0!=0 or self.a1!=0 or self.a2!=0 or self.a3!=0 or self.a4!=0 or self.a5!=0:
            # gamma_tot += gamma_cosk_coskpi4_func(kx, ky, kz)
            # gamma_tot += gamma_poly_func(kx, ky, kz)
            gamma_tot += self.gamma_sinn_cosm(kx, ky, kz)
            # gamma_tot += gamma_tanh_func(kx, ky, kz)
            # gamma_tot += gamma_ndlsco_tl2201_func(kx, ky, kz)
        if self.gamma_kpi4!=0:
            gamma_tot += self.gamma_coskpi4_func(kx, ky, kz)
        if self.gamma_k!=0:
            gamma_tot += self.gamma_k_func(kx, ky, kz)
        if self.gamma_step!=0:
            gamma_tot += self.gamma_step_func(kx, ky, kz)
        if self.gamma_dos_max!=0:
            gamma_tot += self.gamma_DOS_func(vx, vy, vz)
        if self.gamma_cmfp_max!=0:
            gamma_tot += self.gamma_cmfp_func(vx, vy, vz)
        if self.l_path != 0:
            gamma_tot += self.gamma_vF_func(vx, vy, vz)
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

        bands = bObj.e_3D_func(kxx, kyy, kz)
        contours = measure.find_contours(bands, 0)

        gamma_max_list = []
        gamma_min_list = []
        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / bObj.a
            ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / bObj.b
            vx, vy, vz = bObj.v_3D_func(kx, ky, kz)

            gamma_kz = 1 / self.tau_total_func(kx, ky, kz, vx, vy, vz)
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

        bands = bObj.e_3D_func(kxx, kyy, kz)
        contours = measure.find_contours(bands, 0)

        for contour in contours:
            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / bObj.a
            ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / bObj.b
            vx, vy, vz = bObj.v_3D_func(kx, ky, kz)

            gamma_kz = 1 / self.tau_total_func(kx, ky, kz, vx, vy, vz)

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
                            bObj.e_3D_func(kxx, kyy, - 2*pi / bObj.c), 0, colors = '#FF0000', linewidths = 3)
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
        bObj = self.bandObject
        # (1,1) means one plot, and figsize is w x h in inch of figure
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
        # adjust the box of axes regarding the figure size
        fig.subplots_adjust(left=0.15, right=0.25, bottom=0.18, top=0.95)
        axes.remove()

        # Band name
        fig.text(0.72, 0.92, bObj.band_name, fontsize=20, color='#00d900')
        try:
            bObj._band_params["M"]
            fig.text(0.41, 0.92, "AF", fontsize=20,
                        color="#FF0000")
        except:
            None

        # Band Formulas
        fig.text(0.45, 0.445, "Band formula", fontsize=16,
                    color='#008080')
        fig.text(0.45, 0.4, r"$a$ = " + "{0:.2f}".format(bObj.a) + r" $\AA$,  " +
                            r"$b$ = " + "{0:.2f}".format(bObj.b) + r" $\AA$,  " +
                            r"$c$ = " + "{0:.2f}".format(bObj.c) + r" $\AA$", fontsize=12)

        # r"$c$ " + "{0:.2f}".format(bObj.c)
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
            bObj._band_params["M"]
            if bObj.electronPocket == True:
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
        fig.text(0.45, 0.92, "Band Parameters", fontsize=16,
                    color='#008080')
        label_parameters = [r"t = " + "{0:.1f}".format(bObj.energy_scale) + " meV"] +\
                        [key + " = " + "{0:+.3f}".format(value) + r" $t$" for (key, value) in sorted(bObj._band_params.items()) if key!="t"]

        try:  # if it is a AF band
            bObj._band_params["M"]
            label_parameters.append(
                r"$\Delta_{\rm AF}$ =  " + "{0:+.3f}".format(bObj._band_params["M"]) + r"   $t$")
        except:
            None

        h_label = 0.88
        for label in label_parameters:
            fig.text(0.45, h_label, label, fontsize=12)
            h_label -= 0.035

        # Scattering parameters
        fig.text(0.72, 0.86, "Scattering Parameters",
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
        h_label = 0.82
        for label in label_parameters:
            fig.text(0.72, h_label, label, fontsize=12)
            h_label -= 0.035

        ## Inset FS ///////////////////////////////////////////////////////////#
        a = bObj.a
        b = bObj.b
        c = bObj.c

        mesh_graph = 201
        kx = np.linspace(-pi/a, pi/a, mesh_graph)
        ky = np.linspace(-pi/b, pi/b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing='ij')

        axes_FS = plt.axes([-0.02, 0.56, .4, .4])
        axes_FS.set_aspect(aspect=1)
        axes_FS.contour(kxx, kyy, bObj.e_3D_func(
            kxx, kyy, 0), 0, colors='#FF0000', linewidths=1)
        axes_FS.contour(kxx, kyy, bObj.e_3D_func(
            kxx, kyy, pi / c), 0, colors='#00DC39', linewidths=1)
        axes_FS.contour(kxx, kyy, bObj.e_3D_func(
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
        kx_a = np.linspace(-pi / bObj.a, pi / bObj.a,
                        mesh_xy)
        ky_a = np.linspace(-pi / bObj.b, pi / bObj.b,
                        mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = bObj.e_3D_func(kxx, kyy, 0)
        contours = measure.find_contours(bands, 0)

        gamma_max_list = []
        gamma_min_list = []
        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] /
                (mesh_xy - 1) - 0.5) * 2 * pi / bObj.a
            ky = (contour[:, 1] /
                (mesh_xy - 1) - 0.5) * 2 * pi / bObj.b
            vx, vy, vz = bObj.v_3D_func(kx, ky, np.zeros_like(kx))

            gamma_kz0 = 1 / self.tau_total_func(kx, ky, 0, vx, vy, vz)
            gamma_max_list.append(np.max(gamma_kz0))
            gamma_min_list.append(np.min(gamma_kz0))

            points = np.array([kx * bObj.a,
                            ky * bObj.b]).T.reshape(-1, 1, 2)
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
