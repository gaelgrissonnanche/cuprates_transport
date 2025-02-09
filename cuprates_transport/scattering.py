import numpy as np
from numpy import cos, sin, pi, sqrt, arctan2
from scipy.constants import Boltzmann, hbar, elementary_charge, physical_constants
from copy import deepcopy
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Units ////////
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 1e-12 # 1 ps in seconds

## Constant //////
e = elementary_charge # C
kB = Boltzmann # J / K
kB = kB / meV # meV / K

## Decorator
def scattering_method(func):
    """Decorator to mark a method as a scattering function."""
    func.is_scattering_method = True
    return func

class Scattering:
    def __init__(self, scattering_models, scattering_params, **kwargs):

        self.scattering_models = scattering_models
        self._scattering_params = deepcopy(scattering_params)
        self.gamma_tot = np.empty(1) # [ps^-1] total scattering rate array

        # Set scattering rate model parameters as class attributes
        for param, value in self._scattering_params.items():
            setattr(self, param, value)  # Set each parameter as an attribute

        ## Phi angle
        self.phi = None # [radians] angle between the field and the a-axis

        # # Scattering rate energy-dependent
        # self.a_epsilon     = a_epsilon # in THz/meV
        # self.a_abs_epsilon = a_abs_epsilon # in THz/meV
        # self.a_epsilon_2   = a_epsilon_2 # in THz/meV^2
        # self.a_T           = a_T # unitless
        # self.a_T2          = a_T2 # unitless

        # # Scattering rate asymmetric Antoine Georger
        # self.a_asym = a_asym # in THz/meV
        # self.p_asym = p_asym # unitless

        # # Scattering rate k-dependent
        # self.gamma_cmfp_max = gamma_cmfp_max
        # self.gamma_step    = gamma_step
        # self.phi_step      = np.deg2rad(phi_step)
        # self.factor_arcs   = factor_arcs # factor * gamma_0 outsite AF FBZ
        # self.a0 = a0
        # self.a1 = a1
        # self.a2 = a2
        # self.a3 = a3
        # self.a4 = a4
        # self.a5 = a5
        # self.gamma_kpi4 = gamma_kpi4
        # self.powerpi4 = powerpi4
        # self.l_path = l_path # in Angstrom, mean free path for gamma_vF

    # # Special Method >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    # def __setitem__(self, key, value):
    #     # Add security not to add keys later
    #     if key not in self._scattering_params.keys():
    #         print(key + " was not added (new scattering parameters are only"
    #               " allowed within object initialization)")
    #     else:
    #         self._scattering_params[key] = value

    # def __getitem__(self, key):
    #     try:
    #         assert self._scattering_params[key]
    #     except KeyError:
    #         print(key + " is not a defined scattering parameter")
    #     else:
    #         return self._scattering_params[key]

    # Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    def _get_scattering_params(self):
        return self._scattering_params

    def _set_scattering_params(self, scattering_params):
        self._scattering_params = scattering_params
        # Set scattering rate model parameters as class attributes
        for param, value in self._scattering_params.items():
            setattr(self, param, value)  # Set each parameter as an attribute
    scattering_params = property(_get_scattering_params, _set_scattering_params)


    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    def phi_func(self, kx, ky, kz):
        """
        To calcute the phi angle between the field and the a-axis
        """
        ## Make sure kx and ky are in the FBZ to compute Phi.
        # if self.phi is None: # avoid rerunning it if it has been before
        a, b = self.bandObject.a, self.bandObject.b
        kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
        ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
        self.phi = arctan2(ky, kx)
        return self.phi

    ## Scatterint rate models ---------------------------------------------------#
    @scattering_method
    def isotropic(self, kx, ky, kz):
        """
        Scattering rate function
        gamma = gamma_0 for all k states:
        - gamma_0 [ps^-1]
        """
        return self.gamma_0 * np.ones_like(kx)

    @scattering_method
    def cos2phi(self, kx, ky, kz):
        """
        Scattering rate function
        gamma = gamma_k * |cos(2*phi)|^power:
        - gamma_k [ps^-1]
        - power [unitless]
        """
        phi = self.phi_func(kx, ky, kz)
        return self.gamma_k * np.abs(cos(2*phi))**self.power

    @scattering_method
    def cos2phi_coskz(self, kx, ky, kz):
        """
        Scattering rate function
        gamma = gamma_k * |cos(2*phi)|^power:
        - gamma_k [ps^-1]
        - power [unitless]
        """
        a, b, c = self.bandObject.a, self.bandObject.b, self.bandObject.c
        phi = self.phi_func(kx, ky, kz)
        return self.gamma_k * np.abs(cos(2*phi))**self.power * np.abs(cos(kz*c/2))**self.power_z

    @scattering_method
    def sin2phi(self, kx, ky, kz):
        """
        Scattering rate function
        gamma = gamma_k_sin * |sin(2*phi)|^power_sin:
        - gamma_k_sin [ps^-1]
        - power_sin [unitless]
        """
        phi = self.phi_func(kx, ky, kz)
        return self.gamma_k_sin * np.abs(sin(2*phi))**self.power_sin

    @scattering_method
    def gamma_dos_func(self, vx, vy, vz):
        """
        Scattering rate function
        gamma = gamma_dos_max * ( dos_k / dos_k_max):
        - gamma_dos_max [ps^-1]
        Reminder: (v_F_max / v_F) = ( dos_k / dos_k_max)
        """
        dos_max = np.max(self.bandObject.dos_k)
        # value to normalize the DOS to a quantity without units
        return self.gamma_dos_max * (self.bandObject.dos_k / dos_max)

    # @scattering_method
    # def gamma_cmfp_func(self, vx, vy, vz):
    #     vf = sqrt( vx**2 + vy**2 + vz**2 )
    #     vf_max = np.max(self.bandObject.vf)
    #     # value to normalize the DOS to a quantity without units
    #     return self.gamma_cmfp_max * (vf / vf_max)

    # @scattering_method
    # def gamma_vF_func(self, vx, vy, vz):
    #     """vx, vy, vz are in Angstrom.meV
    #     l_path is in Angstrom
    #     """
    #     vF = sqrt(vx**2 + vy**2 + vz**2) / Angstrom * 1e-12 # in Angstrom / ps
    #     return vF / self.l_path

    # @scattering_method
    # def gamma_poly_func(self, kx, ky, kz):
    #     ## Make sure kx and ky are in the FBZ to compute Phi.
    #     a, b = self.bandObject.a, self.bandObject.b
    #     kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    #     ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    #     phi = arctan2(ky, kx)
    #     phi_p = np.abs((np.mod(phi, pi/2)-pi/4))
    #     return (self.a0 + np.abs(self.a1 * phi_p + self.a2 * phi_p**2 +
    #             self.a3 * phi_p**3 + self.a4 * phi_p**4 + self.a5 * phi_p**5))

    # @scattering_method
    # def gamma_tanh_func(self, kx, ky, kz):
    #     ## Make sure kx and ky are in the FBZ to compute Phi.
    #     a, b = self.bandObject.a, self.bandObject.b
    #     kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    #     ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    #     phi = arctan2(ky, kx)
    #     return self.a0 / np.abs(np.tanh(self.a1 + self.a2 * np.abs(cos(2*(phi+pi/4)))**self.a3))

    # @scattering_method
    # def gamma_step_func(self, kx, ky, kz):
    #     ## Make sure kx and ky are in the FBZ to compute Phi.
    #     a, b = self.bandObject.a, self.bandObject.b
    #     kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    #     ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    #     phi = arctan2(ky, kx)
    #     index_low = ((np.mod(phi, pi/2) >= (pi/4 - self.phi_step)) *
    #                 (np.mod(phi, pi/2) <= (pi/4 + self.phi_step)))
    #     index_high = np.logical_not(index_low)
    #     gamma_step_array = np.zeros_like(phi)
    #     gamma_step_array[index_high] = self.gamma_step
    #     return gamma_step_array

    # @scattering_method
    # def gamma_skew_marginal_fl(self, epsilon):
    #     return np.sqrt((self.a_epsilon * epsilon +
    #                     self.a_abs_epsilon * np.abs(epsilon))**2 +
    #                     (self.a_T * kB * meV / hbar * 1e-12 * self.T)**2)

    # @scattering_method
    # def gamma_fl(self, epsilon):
    #     return (self.a_epsilon_2 * epsilon**2 +
    #             self.a_T2 * (kB * meV / hbar * 1e-12 * self.T)**2)

    # @scattering_method
    # def gamma_skew_planckian(self, epsilon):
    #     x = epsilon / (kB * self.T)
    #     x = np.where(x == 0, 1.0e-20, x)
    #     return ((self.a_asym * kB * self.T) * ((x + self.p_asym)/2) * np.cosh(x/2) /
    #             np.sinh((x + self.p_asym)/2) / np.cosh(self.p_asym/2))

    # @scattering_method
    # def factor_arcs_func(self, kx, ky, kz):
    #     # line ky = kx + pi
    #     d1 = ky * self.bandObject.b - kx * self.bandObject.a - pi  # line ky = kx + pi
    #     d2 = ky * self.bandObject.b - kx * self.bandObject.a + pi  # line ky = kx - pi
    #     d3 = ky * self.bandObject.b + kx * self.bandObject.a - pi  # line ky = -kx + pi
    #     d4 = ky * self.bandObject.b + kx * self.bandObject.a + pi  # line ky = -kx - pi

    #     is_in_FBZ_AF = np.logical_and((d1 <= 0)*(d2 >= 0), (d3 <= 0)*(d4 >= 0))
    #     is_out_FBZ_AF = np.logical_not(is_in_FBZ_AF)
    #     factor_out_of_FBZ_AF = np.ones_like(kx)
    #     factor_out_of_FBZ_AF[is_out_FBZ_AF] = self.factor_arcs
    #     return factor_out_of_FBZ_AF


    # Calculates the total scattering rate using Matthiessen's rule -------------#
    def tau_tot_func(self, kx, ky, kz, vx, vy, vz, epsilon = 0):
        """Computes the total lifetime based on the input model
        for the scattering rate"""
        self.gamma_tot = np.zeros_like(kx)
        # Execute each method in scattering_params
        for model in self.scattering_models:
            func = getattr(self, model)
            func_args = func.__code__.co_varnames
            if "epsilon" in func_args:
                self.gamma_tot += func(epsilon)
            elif "vx" in func_args:
                self.gamma_tot += func(vx, vy, vz)
            else:
                self.gamma_tot += func(kx, ky, kz)
        return 1 / self.gamma_tot


if __name__=="__main__":
    scattering_models = ["isotropic", "cos2phi"]
    scattering_params = {"gamma_0": 2, "gamma_k":10, "power":12}
    scattObject = Scattering(scattering_models, scattering_params)
    print(scattObject.gamma_0)
    scattObject.gamma_0 = 10
    print(scattObject.gamma_0)
    print(scattObject.scattering_params)
    scattObject.scattering_params = {"gamma_0": 5, "gamma_k":10, "power":12}
    print(scattObject.gamma_0)
