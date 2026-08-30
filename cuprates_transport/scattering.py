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
    """Mark a method as a selectable scattering-rate model.

    The decorator leaves the callable unchanged except for attaching the
    boolean attribute ``is_scattering_method``.
    """
    func.is_scattering_method = True
    return func

class Scattering:
    """Composable quasiparticle scattering-rate model.

    Parameters
    ----------
    scattering_models : sequence of str
        Method names evaluated and added as rates according to Matthiessen's
        rule. Built-in choices include ``isotropic``, ``cos2phi``,
        ``cos2phi_coskz``, ``sin2phi``, ``gamma_dos_func``, ``polynomial``, and
        ``gamma_skew_marginal_fl``.
    scattering_params : dict
        Parameters required by the selected models. Each item is also exposed
        as an instance attribute.
    **kwargs
        Accepted for cooperative construction from larger parameter mappings.

    Notes
    -----
    Scattering rates are expressed in ``ps^-1`` (numerically THz) and lifetimes
    in ps. ``Conductivity`` subclasses this class and supplies ``bandObject``
    and ``T``, which are needed by some models.
    """
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
        """Return the in-plane momentum angle inside the first Brillouin zone.

        ``kx`` and ``ky`` are wrapped into their first-zone intervals before
        evaluating ``arctan2(ky, kx)``. ``kz`` is accepted for the common model
        signature but is not used. The result, in radians, is cached as
        ``phi``.
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
        """Return the constant rate ``gamma_0`` in ``ps^-1``.

        The output has the shape of ``kx``; ``ky`` and ``kz`` are accepted for
        interface consistency.
        """
        return self.gamma_0 * np.ones_like(kx)

    @scattering_method
    def cos2phi(self, kx, ky, kz):
        """Return ``gamma_k * |cos(2*phi)|**power`` in ``ps^-1``.

        ``gamma_k`` is the antinodal amplitude and ``power`` controls angular
        sharpness. ``phi`` is computed after wrapping momenta into the first
        Brillouin zone.
        """
        phi = self.phi_func(kx, ky, kz)
        return self.gamma_k * np.abs(cos(2*phi))**self.power

    @scattering_method
    def cos2phi_coskz(self, kx, ky, kz):
        """Return a ``cos(2*phi)`` rate modulated along ``kz``.

        The model is ``gamma_k * |cos(2*phi)|**power *
        |cos(kz*c/2)|**power_z`` in ``ps^-1``.
        """
        a, b, c = self.bandObject.a, self.bandObject.b, self.bandObject.c
        phi = self.phi_func(kx, ky, kz)
        return self.gamma_k * np.abs(cos(2*phi))**self.power * np.abs(cos(kz*c/2))**self.power_z

    @scattering_method
    def sin2phi(self, kx, ky, kz):
        """Return ``gamma_k_sin * |sin(2*phi)|**power_sin`` in ``ps^-1``.

        The required amplitude and power are read from ``scattering_params``;
        momenta are wrapped before evaluating the in-plane angle.
        """
        phi = self.phi_func(kx, ky, kz)
        return self.gamma_k_sin * np.abs(sin(2*phi))**self.power_sin

    @scattering_method
    def gamma_dos_func(self, vx, vy, vz):
        """Return a rate proportional to the local Fermi-surface DOS.

        The model is ``gamma_dos_max * dos_k / max(dos_k)`` in ``ps^-1``.
        Velocity arguments are accepted so :meth:`tau_tot_func` recognizes
        this as a velocity-based model, but the cached ``bandObject.dos_k`` is
        used directly.
        """
        dos_max = np.max(self.bandObject.dos_k)
        # value to normalize the DOS to a quantity without units
        return self.gamma_dos_max * (self.bandObject.dos_k / dos_max)

    @scattering_method
    def polynomial(self, kx, ky, kz):
        """Return an absolute fifth-order polynomial in in-plane angle.

        The implemented rate is ``|a1*phi + a2*phi**2 + ... + a5*phi**5|`` in
        ``ps^-1``. Coefficients must be provided through ``scattering_params``.
        """
        phi = self.phi_func(kx, ky, kz)
        return np.abs(self.a1 * phi + self.a2 * phi**2 + self.a3 * phi**3 +
                      self.a4 * phi**4 + self.a5 * phi**5)

    @scattering_method
    def gamma_skew_marginal_fl(self, epsilon):
        """Return the skewed marginal-Fermi-liquid rate.

        Parameters
        ----------
        epsilon : array_like
            Energy relative to the chemical potential in meV.

        Returns
        -------
        ndarray or scalar
            Rate in ``ps^-1`` combining asymmetric linear-energy terms and a
            temperature term controlled by ``a_epsilon``, ``a_abs_epsilon``,
            and ``a_T``.
        """
        return np.sqrt((self.a_epsilon * epsilon +
                        self.a_abs_epsilon * np.abs(epsilon))**2 +
                        (self.a_T * kB * meV / hbar * 1e-12 * self.T)**2)



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
    # def gamma_fl(self, epsilon):
    #     return (self.a_epsilon_2 * epsilon**2 +
    #             self.a_T2 * (kB * meV / hbar * 1e-12 * self.T)**2)



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
        """Evaluate the total lifetime using Matthiessen's rule.

        Parameters
        ----------
        kx, ky, kz : array_like
            Momenta in inverse angstrom.
        vx, vy, vz : array_like
            Group velocities in m/s. They are used only by models whose method
            signature requests velocities.
        epsilon : float or array_like, optional
            Energy in meV for energy-dependent models.

        Returns
        -------
        ndarray
            Total lifetime in ps, with the broadcast shape of the selected
            model outputs.

        Notes
        -----
        The summed rate is stored in ``gamma_tot`` in ``ps^-1``. Model dispatch
        is based on the selected method's argument names.
        """
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
