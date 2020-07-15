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
    def __init__(self,
                 a, b, c,
                 energy_scale,
                 band_params={"t": 1, "tp":-0.136, "tpp":0.068, "tz":0.07, "mu":-0.83},
                 band_name="band_1",
                 epsilon_xy = "", epsilon_z = "",
                 res_xy=20, res_z=7,
                 **trash):

        self._energy_scale = energy_scale  # the value of "t" in meV
        self.a = a  # in Angstrom
        self.b = b  # in Angstrom
        self.c = c  # in Angstrom

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
        self.epsilon_sym = None # intialize this attribute
        if epsilon_xy=="":
            self.epsilon_xy_sym = sp.sympify("- 2*t*(cos(a*kx) + cos(b*ky))" +\
                                             "- 4*tp*cos(a*kx)*cos(b*ky)" +\
                                             "- 2*tpp*(cos(2*a*kx) + cos(2*b*ky))")
        else:
            self.epsilon_xy_sym = sp.sympify(epsilon_xy)

        ## Build the symbolic out-of-plane dispersion
        if epsilon_z=="":
            self.epsilon_z_sym = sp.sympify("- 2*tz*(cos(a*kx) - cos(b*ky))**2*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)")
        else:
            self.epsilon_z_sym = sp.sympify(epsilon_z)

        ## Create the dispersion and velocity functions
        self.e_3D_v_3D_definition()

        ## Discretization
        self.res_xy_rough = 501 # number of subdivisions of the FBZ in units of Pi in the plane for to run the Marching Square
        self.res_xy       = res_xy  # number of subdivisions of the FBZ in units of Pi in the plane for the Fermi surface
        if res_z % 2 == 0:  # make sure it is an odd number
            res_z += 1
        self.res_z = res_z  # number of subdivisions of the FBZ in units of Pi in the plane
        self.half_FS = True # if True, kz 0 -> 2pi, if False, kz -2pi to 2pi

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
        self.discretize_FS(epsilon=epsilon)
        self.dos_k_func()
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
        self.epsilon_sym = self.epsilon_xy_sym + self.epsilon_z_sym  - mu

        ## Velocity /////////////////////////////////////////////////////////////
        self.v_sym = [sp.diff(self.epsilon_sym, kx),
                      sp.diff(self.epsilon_sym, ky),
                      sp.diff(self.epsilon_sym, kz)]

        ## Lambdafity ///////////////////////////////////////////////////////////
        epsilon_func = sp.lambdify(self.var_sym, self.epsilon_sym, 'numpy')
        v_func = sp.lambdify(self.var_sym, self.v_sym, 'numpy')

        ## Numba ////////////////////////////////////////////////////////////////
        self.epsilon_func = jit(epsilon_func, nopython=True, parallel=True)
        self.v_func = jit(v_func, nopython=True, parallel=True)


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
        dks = self.dks / Angstrom # in m^-1
        vf = self.vf * meV * Angstrom # in Joule.m (because in the code vf is not divided by hbar)
        vf_perp = sqrt(vf[0, :]**2 + vf[1, :]**2)  # vf perp to B, in Joule.m
        prefactor = (hbar)**2 / (2 * pi) / self.res_z # divide by the number of kz to average over all kz
        self.mc = prefactor * np.sum(dks / vf_perp) / m0

    def dispersionMesh(self, resX=500, resY=500, resZ=11):
        kx_a = np.linspace(-pi / self.a, pi / self.a, resX)
        ky_a = np.linspace(-pi / self.b, pi / self.b, resY)
        kz_a = np.linspace(-2 * pi / self.c, 2 * pi / self.c, resZ)
        kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a, indexing='ij')
        epsilon = self.e_3D_func(kxx, kyy, kzz)
        return epsilon

    def updateFilling(self, resX=500, resY=500, resZ=11):
        epsilon = self.dispersionMesh(resX, resY, resZ)
        kVolume = epsilon.shape[0] * epsilon.shape[1] * epsilon.shape[2]
        self.n = 2 * np.sum(np.greater_equal(0, epsilon)) / kVolume / self.numberOfBZ # 2 is for the spin
        self.p = 1 - self.n
        return self.n

    def doping(self, resX=500, resY=500, resZ=11, printDoping=False):
        self.updateFilling(resX,resY,resZ)
        if printDoping==True:
            print(self.band_name + " :: " + "p=" + "{0:.3f}".format(self.p))
        return self.p

    def filling(self, resX=500, resY=500, resZ=11):
        self.updateFilling(resX,resY,resZ)
        print("n = " + "{0:.3f}".format(self.n))
        return self.n

    def dopingPerkz(self, resX=500, resY=500, resZ=11):
        epsilon = self.dispersionMesh(resX, resY, resZ)
        # Number of k in the Brillouin zone per plane
        Nz = epsilon.shape[0] * epsilon.shape[1]
        # Number of electron in the Brillouin zone per plane
        n_per_kz = 2 * np.sum(np.greater_equal(0, epsilon), axis=(0, 1)) / Nz / self.numberOfBZ # 2 is for the spin
        p_per_kz = 1 - n_per_kz
        return p_per_kz

    def diffDoping(self, mu, ptarget):
        self._band_params["mu"] = mu
        return self.doping() - ptarget

    def setMuToDoping(self, pTarget, ptol=0.001):
        self._band_params["mu"] = optimize.brentq(self.diffDoping, -10, 10, args=(pTarget,), xtol=ptol)

    def discretize_FS(self, epsilon=0, PrintEnding=False):
        """
        res_xy_rough: make denser rough meshgrid to interpolate after
        """

        kx_a = np.linspace(0, pi / self.a, self.res_xy_rough)
        ky_a = np.linspace(0, pi / self.b, self.res_xy_rough)

        if self.half_FS==True:
            kz_a = np.linspace(0, 2 * pi / self.c, self.res_z)
            # half of FBZ, 2*pi/c because bodycentered unit cell
            dkz = 2 * (2 * pi / self.c / self.res_z) # integrand along z, in A^-1
            # factor 2 for dkz is because integratation is only over half kz,
            # so the final integral needs to be multiplied by 2.
        else:
            kz_a = np.linspace(-2 * pi / self.c, 2 * pi / self.c, self.res_z)
            dkz = 4 * pi / self.c / self.res_z # integrand along z, in A^-1

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        for j, kz in enumerate(kz_a):
            contours = measure.find_contours(self.e_3D_func(kxx, kyy, kz), epsilon)
            number_of_points_per_kz = 0

            for i, contour in enumerate(contours):

                # Contour come in units proportionnal to size of meshgrid
                # one want to scale to units of kx and ky
                x = contour[:, 0] / (self.res_xy_rough - 1) * pi
                y = contour[:, 1] / (self.res_xy_rough - 1) * pi / (self.b / self.a)  # anisotropy

                ds = sqrt(np.diff(x)**2 + np.diff(y)**2)  # segment lengths
                s = np.zeros_like(x)  # arrays of zeros
                s[1:] = np.cumsum(ds)  # integrate path, s[0] = 0

                number_of_points_on_contour = int(max(np.ceil(np.max(s) / (pi/self.res_xy)), 4)) # choose at least a minimum of 4 points per contour
                number_of_points_per_kz += number_of_points_on_contour

                dks = np.max(s) / (number_of_points_on_contour + 1) / self.a  # dk path

                # regular spaced path, add one
                s_int = np.linspace(0, np.max(s), number_of_points_on_contour + 1)
                # interpolate and remove the last point (not to repeat)
                x_int = np.interp(s_int, s, x)[:-1]
                y_int = np.interp(s_int, s, y)[:-1]

                # Rotate the contour to get the entire Fermi surface
                # ### WARNING NOT ROBUST IN THE CASE OF C4 SYMMETRY BREAKING
                x_dump = x_int
                y_dump = y_int
                for angle in [pi / 2, pi, 3 * pi / 2]:
                    x_int_p, y_int_p = self.rotation(x_int, y_int, angle)
                    x_dump = np.append(x_dump, x_int_p)
                    y_dump = np.append(y_dump, y_int_p)
                x_int = x_dump
                y_int = y_dump

                # Put in an array /////////////////////////////////////////////////////#
                if i == 0 and j == 0:  # for first contour and first kz
                    kxf = x_int / self.a
                    kyf = y_int / self.a
                    # self.a (and not b) because anisotropy is taken into account earlier
                    kzf = kz * np.ones_like(x_int)
                    self.dks = dks * np.ones_like(x_int)
                    self.dkz = dkz * np.ones_like(x_int)
                    self.dkf = dks * dkz * np.ones_like(x_int)
                else:
                    kxf = np.append(kxf, x_int / self.a)
                    kyf = np.append(kyf, y_int / self.a)
                    kzf = np.append(kzf, kz * np.ones_like(x_int))
                    self.dks = np.append(self.dks, dks * np.ones_like(x_int))
                    self.dkz = np.append(self.dkz, dkz * np.ones_like(x_int))
                    self.dkf = np.append(self.dkf, dks * dkz * np.ones_like(x_int))

            # discretize one fourth of FS, therefore need * 4
            self.number_of_points_per_kz_list.append(4 * number_of_points_per_kz)

        # dim -> (n, i0) = (xyz, position on FS)
        self.kf = np.vstack([kxf, kyf, kzf])

        # Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0, :], self.kf[1, :], self.kf[2, :])
        # dim -> (n, i0) = (xyz, position on FS)
        self.vf = np.vstack([vx, vy, vz])

        ## Output message
        if PrintEnding == True:
            print("Band: " + self.band_name + ": discretized")

    def rotation(self, x, y, angle):
        xp = cos(angle) * x + sin(angle) * y
        yp = -sin(angle) * x + cos(angle) * y
        return xp, yp

    def dos_k_func(self):
        """
        Density of State of k
        dos_k = 1 / (|grad(E)|) here = 1 / (|v|), because in the def of vf, hbar = 1
        dos_k in  meV^1 Angstrom^-1
        """
        self.dos_k = 1 / sqrt( self.vf[0,:]**2 + self.vf[1,:]**2 +self.vf[2,:]**2 )

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

    def figDiscretizeFS2D(self, kz = 0, meshXY = 1001):
        """Show Discretized 2D Fermi Surface """
        mesh_graph = meshXY
        kx = np.linspace(-pi / self.a, pi / self.a, mesh_graph)
        ky = np.linspace(-pi / self.b, pi / self.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

        fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6))
        fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91)

        fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
        line = axes.plot(self.kf[0,:self.number_of_points_per_kz_list[0]] * self.a,
                         self.kf[1,:self.number_of_points_per_kz_list[0]] * self.b)
        plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
        axes.quiver(self.kf[0,:self.number_of_points_per_kz_list[0]] * self.a,
                    self.kf[1,:self.number_of_points_per_kz_list[0]] * self.b,
                    self.vf[0,:self.number_of_points_per_kz_list[0]],
                    self.vf[1,:self.number_of_points_per_kz_list[0]],
                    color = 'k')

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
        #//////////////////////////////////////////////////////////////////////#

    def figDiscretizeFS3D(self, show_veloticites = False):
        """Show Discretized 3D Fermi Surface """
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.kf[0,:], self.kf[1,:], self.kf[2,:], color='k', marker='.')
        if show_veloticites == True:
            ax.quiver(self.kf[0,:], self.kf[1,:], self.kf[2,:], self.vf[0,:], self.vf[1,:], self.vf[2,:], length=0.1, normalize=True)
        plt.show()

    def figMultipleFS2D(self, meshXY = 1001, averaged_kz_FS = False):
        """Show 2D Fermi Surface for different kz"""
        mesh_graph = meshXY
        kx = np.linspace(-4*pi / self.a, 4*pi / self.a, mesh_graph)
        ky = np.linspace(-4*pi / self.b, 4*pi / self.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

        fig, axes = plt.subplots(1, 1, figsize=(8.5, 5.6))
        fig.subplots_adjust(left=0.01, right=0.75, bottom=0.20, top=0.9)

        doping_per_kz = self.dopingPerkz(resZ=5)[2:]
        fig.text(0.63,0.84, r"$k_{\rm z}$ = 0,      $p$ $\in$ $k_{\rm z}$ = " + str(np.round(doping_per_kz[0], 3)), color = "#FF0000", fontsize = 18)
        fig.text(0.63,0.78, r"$k_{\rm z}$ = $\pi/c$,   $p$ $\in$ $k_{\rm z}$ = " + str(np.round(doping_per_kz[1], 3)), color = "#00DC39", fontsize = 18)
        fig.text(0.63,0.72, r"$k_{\rm z}$ = 2$\pi/c$, $p$ $\in$ $k_{\rm z}$ = " + str(np.round(doping_per_kz[2], 3)), color = "#6577FF", fontsize = 18)

        fig.text(0.63,0.3, r"Average over $k_{\rm z}$", fontsize = 18)
        fig.text(0.63,0.24, r"Total $p$ = " + str(np.round(self.doping(), 3)), fontsize = 18)

        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, pi/self.c), 0, colors = '#00DC39', linewidths = 3)
        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 2*pi/self.c), 0, colors = '#6577FF', linewidths = 3)

        ## Averaged FS among all kz
        if averaged_kz_FS == True:
            kz_array = np.linspace(-2*pi/self.c, 2*pi/self.c, 5)
            dump = 0
            for kz in kz_array:
                dump += self.e_3D_func(kxx, kyy, kz)
            axes.contour(kxx*self.a, kyy*self.b, (1/self.res_z)*dump, 0, colors = '#000000', linewidths = 3, linestyles = "dashed")



        axes.set_xlim(-pi, pi)
        axes.set_ylim(-pi, pi)
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
        axes.set_ylabel(r"$k_{\rm y}$", labelpad = -6)

        axes.set_xticks([-pi, 0., pi])
        axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axes.set_yticks([-pi, 0., pi])
        axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
        axes.set_aspect(aspect=1)

        plt.show()
        #//////////////////////////////////////////////////////////////////////////////#




## Antiferromagnetic Reconstruction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
class Pocket(BandStructure):
    def __init__(self, electronPocket=False, reconstruction_3D=False, **kwargs):
        super().__init__(**kwargs)
        self._electronPocket   = electronPocket
        self.reconstruction_3D = reconstruction_3D # if True the reconstruction is over E_2D + E_z, otherwise just E_2D
        self.numberOfBZ        = 2  # number of BZ we intregrate on as we still work on the unreconstructed FBZ

        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        try:
            assert self._band_params["M"]
        except KeyError:
            self._band_params["M"] = 0.5
            print("Warning! 'M' has to be defined; it has been added and set to 0.5")

        # self.var_sym = list(self.var_sym)
        # self.var_sym.append(sp.Symbol('M'))
        # for params in sorted(self._band_params.keys()):
        #     self.var_sym.append(sp.Symbol(params))
        # self.var_sym = tuple(self.var_sym)

        ## Create the dispersion and velocity functions
        self.e_3D_v_3D_AF_definition()

    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def _get_electronPocket(self):
        return self._electronPocket
    def _set_electronPocket(self, electronPocket):
        self._electronPocket = electronPocket
        self.erase_Fermi_surface()
        self.e_3D_v_3D_AF_definition()
    electronPocket = property(_get_electronPocket, _set_electronPocket)

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def e_3D_v_3D_AF_definition(self):

        """Defines with Sympy the dispersion relation and
        symbolicly derives the velocity"""


        ## Symbolic variables ///////////////////////////////////////////////////
        kx = sp.Symbol('kx')
        ky = sp.Symbol('ky')
        kz = sp.Symbol('kz')
        a  = sp.Symbol('a')
        b  = sp.Symbol('b')
        mu = sp.Symbol('mu')
        M  = sp.Symbol('M')

        ## Dispersion //////////////////////////////////////////////////////////
        if self._electronPocket == True:
            sign = 1
        else:
            sign = -1

        if self.reconstruction_3D == True:
            self.epsilon_sym = self.epsilon_xy_sym + self.epsilon_z_sym
        else:
            self.epsilon_sym = self.epsilon_xy_sym

        self.epsilon_AF_sym = 0.5 * (self.epsilon_sym + self.epsilon_sym.subs([(kx, kx+pi/a), (ky, ky+pi/b)])) + \
            sign * sp.sqrt(0.25*(self.epsilon_sym - self.epsilon_sym.subs([(kx, kx+pi/a), (ky, ky+pi/b)]))**2 + M**2)

        if self.reconstruction_3D == False:
            self.epsilon_AF_sym += self.epsilon_z_sym

        self.epsilon_AF_sym += - mu

        ## Velocity ////////////////////////////////////////////////////////////
        self.v_AF_sym = [sp.diff(self.epsilon_AF_sym, kx), sp.diff(self.epsilon_AF_sym, ky), sp.diff(self.epsilon_AF_sym, kz)]

        ## Lambdafity //////////////////////////////////////////////////////////
        epsilon_func = sp.lambdify(self.var_sym, self.epsilon_AF_sym, 'numpy')
        v_func = sp.lambdify(self.var_sym, self.v_AF_sym, 'numpy')

        ## Numba ////////////////////////////////////////////////////////////////
        self.epsilon_func = jit(epsilon_func, nopython=True, parallel=True)
        self.v_func = jit(v_func, nopython=True, parallel=True)


## Functions to compute the doping of a two bands system and more >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
def doping(bandIterable, printDoping=False):
    totalFilling=0
    if printDoping == True:
        print("------------------------------------------------")
    for band in bandIterable:
        band.updateFilling()
        totalFilling += band.n
        if printDoping == True:
            print(band.band_name + ": band filling = " + "{0:.3f}".format(band.n))
    doping = 1-totalFilling
    if printDoping == True:
        print("total hole doping = " + "{0:.3f}".format(doping))
        print("------------------------------------------------")
    return doping

def dopingCondition(mu,ptarget,bandIterable):
    print("mu = " + "{0:.3f}".format(mu))
    for band in bandIterable:
        band.mu = mu
    return doping(bandIterable) - ptarget

def setMuToDoping(bandIterable, pTarget, ptol=0.001):
    print("Computing mu for hole doping = " + "{0:.3f}".format(pTarget))
    mu = optimize.brentq(dopingCondition, -10, 10, args=(pTarget ,bandIterable), xtol=ptol)
    for band in bandIterable:
        band.mu = mu





