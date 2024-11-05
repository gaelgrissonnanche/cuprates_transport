import numpy as np
from numpy import cos, sin, pi, sqrt
from scipy import optimize
from scipy.constants import hbar
import sympy as sp
from numba import jit
from copy import deepcopy

from cuprates_transport.utils import meV, m0, Angstrom
from cuprates_transport.BandStructure.marching_algo import marching_square, marching_cube
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

class BandStructure:
    def __init__(self,
                 a, b, c,
                 energy_scale,
                 band_params={"t": 1, "tp": -0.136, "tpp": 0.068, "tz": 0.07, "mu": -0.83},
                 band_name="band_1",
                 tight_binding="",
                 res_xy=20, res_z=1,
                 parallel = True, march_square=False,
                 **trash):
        """
        Initializes the BandStructure object.
        :param a, b, c: lattice parameters for orthorombic systems
        :param band_params: tight-binding parameters
        :param band_name: name of the band that this object corresponds to
        :param e_xy, e_z: dispersions of the band along xy and z
        :param res_xy, res_z: resolution of the discretization in plane xy
                              or along z
        :param parallel: decides if the run Numba in parallel or not,
                         it increases the speed for
                         a single instance, but it decreases the speed
                         if multiprocessing is running
        """

        self.a, self.b, self.c = a, b, c    # in Angstrom
        self._energy_scale = energy_scale   # the value of "t" in meV

        self._band_params = deepcopy(band_params)
        # all a fraction of the bandwidth
        self.numberOfBZ = 1  # number of BZ we intregrate on
        self.band_name = band_name  # a string to designate the band

        # Build the symbolic variables
        self.var_sym = [sp.Symbol('kx', real=True), sp.Symbol('ky', real=True), sp.Symbol('kz', real=True),
                        sp.Symbol('a', real=True),  sp.Symbol('b', real=True),  sp.Symbol('c', real=True)]
        for params in sorted(self._band_params.keys()):
            self.var_sym.append(sp.Symbol(params, real=True))
        self.var_sym = tuple(self.var_sym)

        # decides if to run Numba in parallel or not, it increases
        # the speed for a single instance, but it decreases the speed
        # if multiprocessing is running
        self.parallel = parallel

        # Create the dispersion and velocity functions
        if tight_binding == "":
             tight_binding= ("- mu - 2*t*(cos(a*kx) + cos(b*ky))" +
                             "- 4*tp*cos(a*kx)*cos(b*ky)" +
                             "- 2*tpp*(cos(2*a*kx) + cos(2*b*ky))" +
                             "- 2*tz*(cos(a*kx) " +
                             "- cos(b*ky))**2*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)")
        self.e_3D_sym = sp.sympify(tight_binding)
        self.e_3D_v_3D_definition()

        # Discretization
        self.res_xy_rough = 501
        # number of subdivisions of the FBZ in units of Pi in the plane for to run the Marching Square
        if res_xy % 2 == 0: res_xy += 1
        if res_z % 2 == 0: res_z += 1

        # number of subdivisions of the FBZ in units of Pi in the plane for the Fermi surface
        self.res_xy = res_xy
        # number of subdivisions of the FBZ in units of Pi out of the plane
        self.res_z = res_z

        self.march_square = march_square
        # whether to use or not the marching square for higher symmetries

        # Fermi surface
        self.erase_Fermi_surface()      # TODO: I would rename "initialize_Fermi_surface"
        self.mass = None    # in * m_e  # Can this also do in erase_Fermi_surface?

    # Special Method >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    def __setitem__(self, key, value):
        # Add security not to add keys later
        if key not in self._band_params.keys():
            print(key + " was not added (new band parameters are only"
                  " allowed within object initialization)")
        else:
            self.erase_Fermi_surface()
            self._band_params[key] = value

    def __getitem__(self, key):
        try:
            assert self._band_params[key]
        except KeyError:
            print(key + " is not a defined band parameter")
        else:
            return self._band_params[key]

    def get_band_param(self, key):
        return self[key]

    def set_band_param(self, key, val):
        self[key] = val

    # Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    def _get_energy_scale(self):
        return self._energy_scale

    def _set_energy_scale(self, energy_scale):
        self._energy_scale = energy_scale
        self.erase_Fermi_surface()
    energy_scale = property(_get_energy_scale, _set_energy_scale)

    # Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    def erase_Fermi_surface(self):
        # Fermi surface
        self.kf = None      # in Angstrom^-1
        self.vf = None      # in m / s
        self.dkf = None     # in Angstrom^-2
        self.dks = None     # in Angstrom^-1
        self.dkz = None     # in Angstrom^-1
        self.dos_k = None   # in Joule^-1 m^-1
        self.dos_epsilon = None     # in meV^-1
        self.p = None       # hole doping, unknown at first
        self.n = None       # band filling (of electron), unknown at first
        self.number_of_points_per_kz_list = []

    def runBandStructure(self, epsilon=0, printDoping=False):
        self.discretize_fermi_surface(epsilon=epsilon)
        self.doping(res_x=self.res_xy*10, res_y=self.res_xy*10, res_z=self.res_z*10, printDoping=printDoping)

    def discretize_fermi_surface(self, epsilon=0):
        if self.march_square is True:
            self.kf, self.dkf = marching_square(self, epsilon)
        else:
            self.kf, self.dkf = marching_cube(self, epsilon)
        # Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0, :], self.kf[1, :], self.kf[2, :])
        # dim -> (n, i0) = (xyz, position on FS)
        self.vf = np.vstack([vx, vy, vz])
        # Density of State of k, dos_k in  Joule^1 m^-1
        # dos_k = 1 / (|grad(E)|) here = 1 / (hbar * |v|),
        self.dos_k = 1 / (hbar * sqrt(self.vf[0,:]**2 + self.vf[1,:]**2 +self.vf[2,:]**2))

    def doping(self, res_x=500, res_y=500, res_z=500, printDoping=False):
        self.update_filling(res_x,res_y,res_z)
        if printDoping is True:
            print(self.band_name + " :: " + "p=" + "{0:.3f}".format(self.p))
        return self.p

    def update_filling(self, res_x=500, res_y=500, res_z=500):
        e_3D, _, _, _, _, _, _ = self.dispersion_grid(res_x, res_y, res_z)
        kVolume = e_3D.shape[0] * e_3D.shape[1] * e_3D.shape[2]
        self.n = 2 * np.sum(np.greater_equal(0, e_3D)) / kVolume / self.numberOfBZ
        # 2 is for the spin
        self.p = 1 - self.n
        return self.n

    def dispersion_grid(self, res_x=500, res_y=500, res_z=500):
        kx_a = np.linspace(-pi / self.a, pi / self.a, res_x)
        ky_a = np.linspace(-pi / self.b, pi / self.b, res_y)
        kz_a = np.linspace(-2*pi / self.c, 2*pi / self.c, res_z)
        # kz_a = np.linspace(-pi / self.c, pi / self.c, res_z)
        kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a, indexing='ij')
        e_3D = self.e_3D_func(kxx, kyy, kzz)
        return e_3D, kxx, kyy, kzz, kx_a, ky_a, kz_a

    def e_3D_v_3D_definition(self):
        """
        Defines with Sympy the dispersion relation and
        symbolicly derives the velocity
        """
        # Symbolic variables //////////////////////////////////////////////////
        kx = sp.Symbol('kx')
        ky = sp.Symbol('ky')
        kz = sp.Symbol('kz')
        # Velocity ////////////////////////////////////////////////////////////
        self.v_sym = [sp.diff(self.e_3D_sym, kx)* meV / hbar * Angstrom,
                      sp.diff(self.e_3D_sym, ky)* meV / hbar * Angstrom,
                      sp.diff(self.e_3D_sym, kz)* meV / hbar * Angstrom] # m / s

        # Check is one of the velocitiy components is "0" ////////////////////
        k_list = ['kx', 'ky', 'kz']
        for i, v in enumerate(self.v_sym):
            if v == 0:
                self.v_sym[i] = "numpy.zeros_like(" + k_list[i] + ")"

        # Lambdafity //////////////////////////////////////////////////////////
        epsilon_func = sp.lambdify(self.var_sym, self.e_3D_sym, 'numpy')
        v_func = sp.lambdify(self.var_sym, self.v_sym, 'numpy')
        # Just in Time Compile with Numba /////////////////////////////////////
        if self.parallel is True:
            self.epsilon_func = jit(epsilon_func, nopython=True, parallel=True)
            self.v_func = jit(v_func, nopython=True, parallel=True)
        else:
            self.epsilon_func = jit(epsilon_func, nopython=True)
            self.v_func = jit(v_func, nopython=True)

    def e_3D_func(self, kx, ky, kz):
        return self.epsilon_func(kx, ky, kz, *self.bandParameters())

    def v_3D_func(self, kx, ky, kz):
        return self.v_func(kx, ky, kz, *self.bandParameters())

    def bandParameters(self):
        abc = [self.a, self.b, self.c]
        val = [value * self.energy_scale for (key, value)
               in sorted(self._band_params.items())]
        return abc + val

    def mass_func(self):
        """
        The cyclotronic mass in units of m0 (the bare electron mass)
        """
        try:
            dks = self.dks / Angstrom # in m^-1
        except TypeError:
            exit("ERROR: self.dks contains NaN. Are you using marching square?")
        vf = self.vf # in m / s
        vf_perp = sqrt(vf[0, :]**2 + vf[1, :]**2)  # vf perp to B, in m / s
        prefactor = hbar / (2 * pi) / self.res_z # divide by the number of kz to average over all kz
        self.mass = prefactor * np.sum(dks / vf_perp) / m0

    def filling(self, res_x=500, res_y=500, res_z=500):
        self.update_filling(res_x,res_y,res_z)
        print("n = " + "{0:.3f}".format(self.n))
        return self.n

    def doping_per_kz(self, res_x=500, res_y=500, res_z=11):
        e_3D, _, _, _, _, _, _ = self.dispersion_grid(res_x, res_y, res_z)
        # Number of k in the Brillouin zone per plane
        Nz = e_3D.shape[0] * e_3D.shape[1]
        # Number of electron in the Brillouin zone per plane
        n_per_kz = 2 * np.sum(np.greater_equal(0, e_3D), axis=(0, 1)) / Nz / self.numberOfBZ
        # 2 is for the spin
        p_per_kz = 1 - n_per_kz
        return n_per_kz, p_per_kz

    def diff_doping(self, mu, p_target):
        self._band_params["mu"] = mu
        return self.doping() - p_target

    def set_mu_to_doping(self, p_target, ptol=0.001):
        self._band_params["mu"] = optimize.brentq(self.diff_doping, -10, 10,
                                                  args=(p_target,), xtol=ptol)

    def rotation(self, x, y, angle):
        xp = cos(angle) * x + sin(angle) * y
        yp = -sin(angle) * x + cos(angle) * y
        return xp, yp

    def dos_epsilon_func(self):
        """
        Density of State integrated at the Fermi level
            dos_epsilon = int( dkf / |grad(E(k))| ) = int( dkf * dos_k )
        Units :
            - dos_epsilon in Joule^-1 m^-3
            - dkf in Angstrom^-2
        """
        prefactor =  2 / (2*pi)**3 # factor 2 for the spin
        self.dos_epsilon = prefactor * np.sum(self.dkf / Angstrom**2 * self.dos_k)
