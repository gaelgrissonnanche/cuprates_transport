import numpy as np
from numpy import cos, sin, pi, sqrt
from scipy import optimize
from scipy.constants import electron_mass, physical_constants
import sympy as sp
from skimage import measure
from numba import jit
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# Constant //////
hbar = 1  # velocity will be in units of 1 / hbar,
# this hbar is taken into accound in the constant units_move_eq

## Units ////////
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule
m0 = electron_mass # in kg
Angstrom = 1e-10  # 1 A in meters


class BandStructure:
    from .bandstructure_plot import figDiscretizeFS3D, figDiscretizeFS2D,\
    figMultipleFS2D

    from .bandstructure_marching import marching_cube, marching_square

    def __init__(self,
                 a, b, c,
                 energy_scale,
                 band_params={"t": 1, "tp":-0.136, "tpp":0.068, "tz":0.07, "mu":-0.83},
                 band_name="band_1",
                 e_xy = "", e_z = "",
                 res_xy=20, res_z=1,
                 parallel=True,
                 **trash):
        """
        Initializes the BandStructure object.
        :param a, b, c: lattice parameters for orthorombic systems
        :param band_params: tight-binding parameters
        :param band_name: name of the band that this object corresponds to
        :param e_xy, e_z: dispersions of the band along xy and z
        :param res_xy, res_z: resolution of the discretization in plane xy or along z
        :param parallel: decides if the run Numba in parallel or not, it increases the speed for
                         a single instance, but it decreases the speed if multiprocessing is running
        """

        self._energy_scale = energy_scale  # the value of "t" in meV
        self.a = a  # in Angstrom
        self.b = b  # in Angstrom
        self.c = c  # in Angstrom

        self.parallel = parallel # decides if to run Numba in parallel or not,
                                 # it increases the speed for a single instance
                                 # but it decreases the speed if multiprocessing
                                 # is running

        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        try:
            assert band_params["t"]==1
        except KeyError:
            band_params["t"] = 1
            print("Warning! 't' has to be defined; it has been added and set to 1")
        except AssertionError:
            band_params["t"] = 1
            print("Warning! 't' has been set to 1, its value must be set in 'energy_scale' in meV")

        try:
            assert band_params["mu"]
        except KeyError:
            band_params["mu"] = 0
            print("Warning! 'mu' has to be defined; it has been added and set to 0")


        self._band_params = deepcopy(band_params) # all a fraction of the bandwidth
        self.numberOfBZ = 1 # number of BZ we intregrate on
        self.band_name = band_name # a string to designate the band

        ## Build the symbolic variables
        self.var_sym = [sp.Symbol('kx'), sp.Symbol('ky'), sp.Symbol('kz'),
                        sp.Symbol('a'),  sp.Symbol('b'),  sp.Symbol('c')]
        for params in sorted(self._band_params.keys()):
            self.var_sym.append(sp.Symbol(params))
        self.var_sym = tuple(self.var_sym)

        ## Build the symbolic in-plane dispersion
        self.e_3D_sym = None # intialize this attribute
        if e_xy=="":
            self.e_xy_sym = sp.sympify("- 2*t*(cos(a*kx) + cos(b*ky))" +\
                                       "- 4*tp*cos(a*kx)*cos(b*ky)" +\
                                       "- 2*tpp*(cos(2*a*kx) + cos(2*b*ky))")
        else:
            e_xy = e_xy.replace("mu", "0") # replace is just to remove "mu" if the user has entered it by mistake
            self.e_xy_sym = sp.sympify(e_xy)

        ## Build the symbolic out-of-plane dispersion
        if e_z=="":
            self.e_z_sym = sp.sympify("- 2*tz*(cos(a*kx) - cos(b*ky))**2*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)")
        else:
            e_z = e_z.replace("mu", "0") # replace is just to remove "mu" if the user has entered it by mistake
            self.e_z_sym = sp.sympify(e_z)

        ## Create the dispersion and velocity functions
        self.e_3D_v_3D_definition()

        ## Discretization
        self.res_xy_rough = 501 # number of subdivisions of the FBZ in units of Pi in the plane for to run the Marching Square
        if res_xy % 2 == 0:
            res_xy += 1
        if res_z % 2 == 0:  # make sure it is an odd number
            res_z += 1
        self.res_xy = res_xy  # number of subdivisions of the FBZ in units of Pi in the plane for the Fermi surface
        self.res_z = res_z  # number of subdivisions of the FBZ in units of Pi in the plane
        self.march_square = False # wether to use or not the marching square for higher symmetries

        ## Fermi surface
        self.kf  = None # in Angstrom^-1
        self.vf  = None # in meV Angstrom (because hbar=1)
        self.dkf = None # in Angstrom^-2
        self.dks = None # in Angstrom^-1
        self.dkz = None # in Angstrom^-1
        self.dos_k  = None # in meV^-1 Angstrom^-1
        self.dos_epsilon = None # in meV^-1
        self.p = None # hole doping, unknown at first
        self.n = None # band filling (of electron), unknown at first

        ## Save number of points in each kz plane
        self.number_of_points_per_kz_list = []

    ## Special Method >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def __setitem__(self, key, value):
        ## Add security not to add keys later
        if key not in self._band_params.keys():
            print(key + " was not added (new band parameters are only allowed within object initialization)")
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

    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def _get_energy_scale(self):
        return self._energy_scale
    def _set_energy_scale(self, energy_scale):
        self._energy_scale = energy_scale
        self.erase_Fermi_surface()
    energy_scale = property(_get_energy_scale, _set_energy_scale)

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def runBandStructure(self, epsilon=0, printDoping=False):
        self.discretize_fermi_surface(epsilon=epsilon)
        self.doping(printDoping=printDoping)

    def erase_Fermi_surface(self):
        self.kf  = None
        self.vf  = None
        self.dkf = None
        self.dks = None
        self.dkz = None
        self.p    = None
        self.n    = None
        self.dos_k  = None
        self.dos_epsilon = None
        self.vf_mean  = None
        self.number_of_points_per_kz_list = []

    def e_3D_v_3D_definition(self):
        """Defines with Sympy the dispersion relation and
        symbolicly derives the velocity"""
        ## Symbolic variables ///////////////////////////////////////////////////
        kx = sp.Symbol('kx')
        ky = sp.Symbol('ky')
        kz = sp.Symbol('kz')
        mu = sp.Symbol('mu')
        ## Dispersion 3D ////////////////////////////////////////////////////////
        self.e_3D_sym = self.e_xy_sym + self.e_z_sym  - mu
        ## Velocity /////////////////////////////////////////////////////////////
        self.v_sym = [sp.diff(self.e_3D_sym, kx),
                      sp.diff(self.e_3D_sym, ky),
                      sp.diff(self.e_3D_sym, kz)]
        ## Lambdafity ///////////////////////////////////////////////////////////
        epsilon_func = sp.lambdify(self.var_sym, self.e_3D_sym, 'numpy')
        v_func = sp.lambdify(self.var_sym, self.v_sym, 'numpy')
        ## Just in Time Compile with Numba ///////////////////////////////////////
        if self.parallel is True:
            self.epsilon_func = jit(epsilon_func, nopython=True, parallel=True)
            self.v_func = jit(v_func, nopython=True, parallel=True)
        else:
            self.epsilon_func = jit(epsilon_func, nopython=True)
            self.v_func = jit(v_func, nopython=True)

    def bandParameters(self):
        return [self.a, self.b, self.c] + [value * self.energy_scale for (key, value) in sorted(self._band_params.items())]

    def e_3D_func(self, kx, ky, kz):
        return self.epsilon_func(kx, ky, kz, *self.bandParameters())

    def v_3D_func(self, kx, ky, kz):
        return self.v_func(kx, ky, kz, *self.bandParameters())

    def mc_func(self):
        """
        The cyclotronic mass in units of m0 (the bare electron mass)
        """
        hbar = 1.05e-34 # m2 kg / s
        try:
            dks = self.dks / Angstrom # in m^-1
        except TypeError:
            exit("ERROR: self.dks contains NaN. Are you using marching square?")
        vf = self.vf * meV * Angstrom # in Joule.m (because in the code vf is not divided by hbar)
        vf_perp = sqrt(vf[0, :]**2 + vf[1, :]**2)  # vf perp to B, in Joule.m
        prefactor = (hbar)**2 / (2 * pi) / self.res_z # divide by the number of kz to average over all kz
        self.mc = prefactor * np.sum(dks / vf_perp) / m0

    def dispersion_grid(self, res_x=500, res_y=500, res_z=11):
        kx_a = np.linspace(-pi / self.a, pi / self.a, res_x)
        ky_a = np.linspace(-pi / self.b, pi / self.b, res_y)
        kz_a = np.linspace(-2 * pi / self.c, 2 * pi / self.c, res_z)
        kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a, indexing='ij')
        e_3D = self.e_3D_func(kxx, kyy, kzz)
        return e_3D, kxx, kyy, kzz, kx_a, ky_a, kz_a

    def update_filling(self, res_x=500, res_y=500, res_z=11):
        e_3D, _, _, _, _, _, _ = self.dispersion_grid(res_x, res_y, res_z)
        kVolume = e_3D.shape[0] * e_3D.shape[1] * e_3D.shape[2]
        self.n = 2 * np.sum(np.greater_equal(0, e_3D)) / kVolume / self.numberOfBZ # 2 is for the spin
        self.p = 1 - self.n
        return self.n

    def doping(self, res_x=500, res_y=500, res_z=11, printDoping=False):
        self.update_filling(res_x,res_y,res_z)
        if printDoping==True:
            print(self.band_name + " :: " + "p=" + "{0:.3f}".format(self.p))
        return self.p

    def filling(self, res_x=500, res_y=500, res_z=11):
        self.update_filling(res_x,res_y,res_z)
        print("n = " + "{0:.3f}".format(self.n))
        return self.n

    def doping_per_kz(self, res_x=500, res_y=500, res_z=11):
        e_3D, _, _, _, _, _, _ = self.dispersion_grid(res_x, res_y, res_z)
        # Number of k in the Brillouin zone per plane
        Nz = e_3D.shape[0] * e_3D.shape[1]
        # Number of electron in the Brillouin zone per plane
        n_per_kz = 2 * np.sum(np.greater_equal(0, e_3D), axis=(0, 1)) / Nz / self.numberOfBZ # 2 is for the spin
        p_per_kz = 1 - n_per_kz
        return n_per_kz, p_per_kz

    def diff_doping(self, mu, p_target):
        self._band_params["mu"] = mu
        return self.doping() - p_target

    def set_mu_to_doping(self, p_target, ptol=0.001):
        self._band_params["mu"] = optimize.brentq(self.diff_doping, -10, 10,
                                                  args=(p_target,), xtol=ptol)

    def discretize_fermi_surface(self, epsilon=0):
        if self.march_square==True:
            self.kf, self.dkf = self.marching_square(epsilon)
        else:
            self.kf, self.dkf = self.marching_cube(epsilon)
        # Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0, :], self.kf[1, :], self.kf[2, :])
        # dim -> (n, i0) = (xyz, position on FS)
        self.vf = np.vstack([vx, vy, vz])
        # Density of State of k, dos_k in  meV^1 Angstrom^-1
        # dos_k = 1 / (|grad(E)|) here = 1 / (|v|), because in the def of vf, hbar = 1
        self.dos_k = 1 / sqrt(self.vf[0,:]**2 + self.vf[1,:]**2 +self.vf[2,:]**2)

    def rotation(self, x, y, angle):
        xp = cos(angle) * x + sin(angle) * y
        yp = -sin(angle) * x + cos(angle) * y
        return xp, yp

    def dos_epsilon_func(self):
        """
        Density of State integrated at the Fermi level
            dos_k = int( dkf / |grad(E(k))| )
        Units :
            - dos_epsilon in meV^-1 Angstrom^-3
            - dkf in Angstrom^-2
        """
        prefactor =  2 / (2*pi)**3 # factor 2 for the spin
        self.dos_epsilon = prefactor * np.sum(self.dkf * self.dos_k)


## Functions to compute the doping of a two bands system and more >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
def doping(bandIterable, printDoping=False):
    totalFilling=0
    if printDoping == True:
        print("------------------------------------------------")
    for band in bandIterable:
        band.update_filling()
        totalFilling += band.n
        if printDoping == True:
            print(band.band_name + ": band filling = " + "{0:.3f}".format(band.n))
    doping = 1-totalFilling
    if printDoping == True:
        print("total hole doping = " + "{0:.3f}".format(doping))
        print("------------------------------------------------")
    return doping

def dopingCondition(mu,p_target,bandIterable):
    print("mu = " + "{0:.3f}".format(mu))
    for band in bandIterable:
        band.mu = mu
    return doping(bandIterable) - p_target

def set_mu_to_doping(bandIterable, p_target, ptol=0.001):
    print("Computing mu for hole doping = " + "{0:.3f}".format(p_target))
    mu = optimize.brentq(dopingCondition, -10, 10, args=(p_target ,bandIterable), xtol=ptol)
    for band in bandIterable:
        band.mu = mu
