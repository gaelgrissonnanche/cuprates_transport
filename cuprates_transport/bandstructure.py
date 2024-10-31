import numpy as np
from numpy import cos, sin, pi, sqrt
from scipy import optimize
from scipy.constants import electron_mass, physical_constants, hbar
import sympy as sp
from numba import jit
from copy import deepcopy
from skimage import measure
import matplotlib as mpl
import matplotlib.pyplot as plt
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

# Units ////////
meV = physical_constants["electron volt"][0] * 1e-3     # 1 meV in Joule
m0 = electron_mass                                      # in kg
Angstrom = 1e-10                                        # 1 A in meters


class BandStructure:
    def __init__(self,
                 a, b, c,
                 energy_scale,
                 band_params={"t": 1, "tp": -0.136, "tpp": 0.068, "tz": 0.07,
                              "mu": -0.83},
                 band_name="band_1",
                 tight_binding=("- mu - 2*t*(cos(a*kx) + cos(b*ky))" +
                    "- 4*tp*cos(a*kx)*cos(b*ky)" +
                    "- 2*tpp*(cos(2*a*kx) + cos(2*b*ky))" +
                    "- 2*tz*(cos(a*kx) " +
                    "- cos(b*ky))**2*cos(a*kx/2)*cos(b*ky/2)*cos(c*kz/2)"),
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

        self._energy_scale = energy_scale   # the value of "t" in meV
        self.a = a                          # in Angstrom
        self.b = b                          # in Angstrom
        self.c = c                          # in Angstrom

        # decides if to run Numba in parallel or not, it increases
        # the speed for a single instance, but it decreases the speed
        # if multiprocessing is running
        self.parallel = parallel

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

        # Create the dispersion and velocity functions
        self.e_3D_sym = sp.sympify(tight_binding)
        self.e_3D_v_3D_definition()

        # Discretization
        self.res_xy_rough = 501
        # number of subdivisions of the FBZ in units of Pi in
        # the plane for to run the Marching Square
        if res_xy % 2 == 0:
            res_xy += 1
        if res_z % 2 == 0:  # make sure it is an odd number
            res_z += 1
        self.res_xy = res_xy
        # number of subdivisions of the FBZ in units of Pi
        # in the plane for the Fermi surface
        self.res_z = res_z
        # number of subdivisions of the FBZ in units of Pi in the plane
        self.march_square = march_square
        # whether to use or not the marching square for higher symmetries

        # Fermi surface
        self.kf = None      # in Angstrom^-1
        self.vf = None      # in m / s
        self.mass = None    # in * m_e
        self.dkf = None     # in Angstrom^-2
        self.dks = None     # in Angstrom^-1
        self.dkz = None     # in Angstrom^-1
        self.dos_k = None   # in Joule^-1 m^-1
        self.dos_epsilon = None     # in meV^-1
        self.p = None       # hole doping, unknown at first
        self.n = None       # band filling (of electron), unknown at first
        self.number_of_points_per_kz_list = []

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
    def runBandStructure(self, epsilon=0, printDoping=False):
        self.discretize_fermi_surface(epsilon=epsilon)
        self.doping(res_x=self.res_xy*10, res_y=self.res_xy*10, res_z=self.res_z*10, printDoping=printDoping)

    def erase_Fermi_surface(self):
        self.kf  = None
        self.vf  = None
        self.dkf = None
        self.dks = None
        self.dkz = None
        self.p   = None
        self.n   = None
        self.dos_k = None
        self.dos_epsilon = None
        self.number_of_points_per_kz_list = []

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

    def bandParameters(self):
        abc = [self.a, self.b, self.c]
        val = [value * self.energy_scale for (key, value)
               in sorted(self._band_params.items())]
        return abc + val

    def e_3D_func(self, kx, ky, kz):
        return self.epsilon_func(kx, ky, kz, *self.bandParameters())

    def v_3D_func(self, kx, ky, kz):
        return self.v_func(kx, ky, kz, *self.bandParameters())

    def mass_func(self):
        # """
        # The effective mass in units of m0 (the bare electron mass)
        # """
        # hbar = 1.05457182e-34     # m2 kg / s
        # kf = self.kf / Angstrom
        # vf = self.vf * meV * Angstrom / hbar
        # # print( meV * Angstrom / hbar)
        # # in Joule.m (because in the code vf is not divided by hbar)
        # kf_norm = sqrt(kf[0, :]**2 + kf[1, :]**2 + kf[2, :]**2)  # in m^-1
        # vf_norm = sqrt(vf[0, :]**2 + vf[1, :]**2 + vf[2, :]**2)  # in Joule.m
        # self.mass = hbar * np.min(kf_norm/vf_norm) / m0

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


    def dispersion_grid(self, res_x=500, res_y=500, res_z=500):
        kx_a = np.linspace(-pi / self.a, pi / self.a, res_x)
        ky_a = np.linspace(-pi / self.b, pi / self.b, res_y)
        kz_a = np.linspace(-2*pi / self.c, 2*pi / self.c, res_z)
        # kz_a = np.linspace(-pi / self.c, pi / self.c, res_z)
        kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a, indexing='ij')
        e_3D = self.e_3D_func(kxx, kyy, kzz)
        return e_3D, kxx, kyy, kzz, kx_a, ky_a, kz_a

    def update_filling(self, res_x=500, res_y=500, res_z=500):
        e_3D, _, _, _, _, _, _ = self.dispersion_grid(res_x, res_y, res_z)
        kVolume = e_3D.shape[0] * e_3D.shape[1] * e_3D.shape[2]
        self.n = 2 * np.sum(np.greater_equal(0, e_3D)) / kVolume / self.numberOfBZ
        # 2 is for the spin
        self.p = 1 - self.n
        return self.n

    def doping(self, res_x=500, res_y=500, res_z=500, printDoping=False):
        self.update_filling(res_x,res_y,res_z)
        if printDoping is True:
            print(self.band_name + " :: " + "p=" + "{0:.3f}".format(self.p))
        return self.p

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

    def discretize_fermi_surface(self, epsilon=0):
        if self.march_square is True:
            self.kf, self.dkf = self.marching_square(epsilon)
        else:
            self.kf, self.dkf = self.marching_cube(epsilon)
        # Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0, :], self.kf[1, :], self.kf[2, :])
        # dim -> (n, i0) = (xyz, position on FS)
        self.vf = np.vstack([vx, vy, vz])
        # Density of State of k, dos_k in  Joule^1 m^-1
        # dos_k = 1 / (|grad(E)|) here = 1 / (hbar * |v|),
        self.dos_k = 1 / (hbar * sqrt(self.vf[0,:]**2 + self.vf[1,:]**2 +self.vf[2,:]**2))

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


    ## Marching algorithms ------------------------------------------------------#


    def marching_cube(self, epsilon=0):
        """
        Marching cube algorithm used to discretize the Fermi surface
        It returns: - kf with dimensions (xyz, position on FS)
                    - dkf with dimensions (position on FS)
        """
        # Generate a uniform meshgrid
        e_3D, _, _, _, kx_a, ky_a, kz_a = self.dispersion_grid(self.res_xy,
            self.res_xy, self.res_z)
        # Use marching cubes to discritize the Fermi surface
        verts, faces, _, _ = measure.marching_cubes(e_3D,
                level=epsilon, spacing=(kx_a[1]-kx_a[0],
                                        ky_a[1]-ky_a[0],
                                        kz_a[1]-kz_a[0]),
                                        method='lewiner')
                                    #---- Shape of verts is: (N_verts, 3)
                                    #---- Shape of faces is: (N_faces, 3)
                                    #---- Shape of triangles is: (N_faces, 3, 3)
                                    #---- Shape of sides is: (N_faces, 2, 3)
                                    #---- Shape of normal_vecs is: (N_faces, 3)
                                    #---- Shape of areas is: (N_faces)
        # Recenter the Fermi surface after Marching Cube in the center of the BZ
        verts[:,0] = verts[:,0] - kx_a[-1]
        verts[:,1] = verts[:,1] - ky_a[-1]
        verts[:,2] = verts[:,2] - kz_a[-1]
        triangles = verts[faces]
        # Calculate areas
        sides = np.diff(triangles, axis=-2)
        # vectors that represent the sides of the triangles
        normal_vecs = np.cross(sides[...,0,:], sides[...,1,:])
        # cross product of two vectors
        # of the faces to calculate the areas of the triangles
        areas = np.linalg.norm(normal_vecs, axis=-1)/2
        # calculate the area of the triangles
        # by taking the norm of the cross product vector and divide by 2

        # Compute weight of each kf in surface integral
        dkf = np.zeros(len(verts))
        verts_repeated = faces.flatten() # shape is (3*N_faces)
        weights = np.repeat(areas/3, 3)  #1/3 for the volume of an irregular triangular prism
        dkf += np.bincount(verts_repeated, weights)
        kf = verts.transpose()
        return kf, dkf

    def marching_square(self, epsilon=0):
        """
        Marching square algorithm used to discretize the Fermi surface per slices
        It returns: - kf with dimensions (xyz, position on FS)
                    - dkf with dimensions (position on FS)
        """
        # Initialize kx and ky arrays
        # res_xy_rough: make denser rough meshgrid to interpolate after
        if self.a == self.b:  # tetragonal case
            kx_a = np.linspace(0, pi / self.a, self.res_xy_rough)
            ky_a = np.linspace(0, pi / self.b, self.res_xy_rough)
        else:  # orthorhombic
            kx_a = np.linspace(-pi / self.a, pi / self.a, 2*self.res_xy_rough)
            ky_a = np.linspace(-pi / self.b, pi / self.b, 2*self.res_xy_rough)
        # Initialize kz array
        kz_a = np.linspace(-2 * pi / self.c, 2 * pi / self.c, self.res_z)
        dkz = kz_a[1] - kz_a[0]  # integrand along z, in A^-1
        # Meshgrid for kx and ky
        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')
        self.number_of_points_per_kz_list = []
        # Loop over the kz array
        for j, kz in enumerate(kz_a):
            contours = measure.find_contours(self.e_3D_func(kxx, kyy, kz), epsilon)
            number_of_points_per_kz = 0
            ## Loop over the different pieces of Fermi surfaces
            for i, contour in enumerate(contours):
                # Contour come in units proportionnal to size of meshgrid
                # one want to scale to units of kx and ky
                if self.a == self.b:
                    x = contour[:, 0] / (self.res_xy_rough - 1) * pi
                    y = contour[:, 1] / (self.res_xy_rough - 1) * pi / (self.b / self.a)  # anisotropy
                else:
                    x = (contour[:, 0] / (2*self.res_xy_rough - 1) - 0.5) * 2*pi
                    y = (contour[:, 1] / (2*self.res_xy_rough - 1) - 0.5) * 2*pi / (self.b / self.a) # anisotropy
                # path
                ds = sqrt(np.diff(x)**2 + np.diff(y)**2)  # segment lengths
                s = np.zeros_like(x)  # arrays of zeros
                s[1:] = np.cumsum(ds)  # integrate path, s[0] = 0

                number_of_points_on_contour = int(max(np.ceil(np.max(s) / (pi/self.res_xy)), 4)) # choose at least a minimum of 4 points per contour
                number_of_points_per_kz += number_of_points_on_contour

                # interpolate and remove the last point (not to repeat)
                s_int = np.linspace(0, np.max(s), number_of_points_on_contour) # path
                dks = (s_int[1]- s_int[0]) / self.a  # dk path
                x_int = np.interp(s_int, s, x)[:-1]
                y_int = np.interp(s_int, s, y)[:-1]
                if self.a == self.b:
                    # For tetragonal symmetry, rotate the contour to get the entire Fermi surface
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
                    kyf = y_int / self.b
                    # self.a (and not b) because anisotropy is taken into account earlier
                    kzf = kz * np.ones_like(x_int)
                    dkf = dks * dkz * np.ones_like(x_int)
                    self.dks = dks * np.ones_like(x_int)
                    self.dkz = dkz * np.ones_like(x_int)
                else:
                    kxf = np.append(kxf, x_int / self.a)
                    kyf = np.append(kyf, y_int / self.b)
                    kzf = np.append(kzf, kz * np.ones_like(x_int))
                    dkf = np.append(dkf, dks * dkz * np.ones_like(x_int))
                    self.dks = np.append(self.dks, dks * np.ones_like(x_int))
                    self.dkz = np.append(self.dkz, dkz * np.ones_like(x_int))
            if self.a == self.b:
                # discretize one fourth of FS, therefore need * 4
                self.number_of_points_per_kz_list.append(4 * number_of_points_per_kz)
            else:
                self.number_of_points_per_kz_list.append(number_of_points_per_kz)
        # dim -> (n, i0) = (xyz, position on FS)
        kf = np.vstack([kxf, kyf, kzf])
        return kf, dkf

    ## Plots --------------------------------------------------------------------#

    # ///// RC Parameters ////// #
    mpl.rcdefaults()
    mpl.rcParams['font.size'] = 24.         # Fontsize
    mpl.rcParams['font.family'] = 'Arial'   # Font Arial
    mpl.rcParams['axes.labelsize'] = 24.
    mpl.rcParams['xtick.labelsize'] = 24
    mpl.rcParams['ytick.labelsize'] = 24
    mpl.rcParams['xtick.direction'] = "in"
    mpl.rcParams['ytick.direction'] = "in"
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.major.width'] = 0.6
    mpl.rcParams['ytick.major.width'] = 0.6
    mpl.rcParams['axes.linewidth'] = 0.6    # Thickness of the axes lines
    mpl.rcParams['pdf.fonttype'] = 3

    def figDiscretizeFS2D(self, kz = 0, meshXY = 1001):
        """
        Show Discretized 2D Fermi Surface.
        """
        try:
            assert self.march_square
        except AssertionError:
            print("'figDiscretizeFS2D' only works for march_square = True")
            return


        mesh_graph = meshXY
        kx = np.linspace(-pi / self.a, pi / self.a, mesh_graph)
        ky = np.linspace(-pi / self.b, pi / self.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

        fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6))
        fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91)

        fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)

        nb_pkz = self.number_of_points_per_kz_list
        nkz = len(nb_pkz)
        npkz0 = np.sum(nb_pkz[:nkz//2])
        npkz1 = npkz0 + nb_pkz[nkz//2]

        line = axes.plot(self.kf[0, npkz0: npkz1] * self.a,
                        self.kf[1, npkz0: npkz1] * self.b)
        # line = axes.plot(self.kf[0,:self.number_of_points_per_kz_list[0]] * self.a,
        #                  self.kf[1,:self.number_of_points_per_kz_list[0]] * self.b)
        plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
        axes.quiver(self.kf[0, npkz0: npkz1] * self.a,
                    self.kf[1, npkz0: npkz1] * self.b,
                    self.vf[0, npkz0: npkz1],
                    self.vf[1, npkz0: npkz1],
                    color = 'k')
        # axes.quiver(self.kf[0,:self.number_of_points_per_kz_list[0]] * self.a,
        #             self.kf[1,:self.number_of_points_per_kz_list[0]] * self.b,
        #             self.vf[0,:self.number_of_points_per_kz_list[0]],
        #             self.vf[1,:self.number_of_points_per_kz_list[0]],
        #             color = 'k')

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

    def figDiscretizeFS3D(self, show_veloticites = False):
        """
        Show Discretized 3D Fermi Surface.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.kf[0,:], self.kf[1,:], self.kf[2,:], color='k', marker='.')
        if show_veloticites == True:
            ax.quiver(self.kf[0,:], self.kf[1,:], self.kf[2,:], self.vf[0,:], self.vf[1,:], self.vf[2,:], length=0.1, normalize=True)
        plt.show()

    def figMultipleFS2D(self, meshXY = 1001, averaged_kz_FS = False):
        """
        Show 2D Fermi Surface for different kz.
        """
        mesh_graph = meshXY
        kx = np.linspace(-4*pi / self.a, 4*pi / self.a, mesh_graph)
        ky = np.linspace(-4*pi / self.b, 4*pi / self.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')
        ## Figure
        fig, axes = plt.subplots(1, 1, figsize=(8.5, 5.6))
        fig.subplots_adjust(left=0.01, right=0.75, bottom=0.20, top=0.9)
        # doping_per_kz returns (n, p), so we look at p here.
        doping_per_kz = self.doping_per_kz(res_z=5)[1][2:]
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


## Functions to compute the doping of a two bands system and more ---------------#

def doping(bandIterable, printDoping=False):
    totalFilling=0
    if printDoping is True:
        print("------------------------------------------------")
    for band in bandIterable:
        band.update_filling()
        totalFilling += band.n
        if printDoping is True:
            print(band.band_name + ": band filling = " + "{0:.3f}".format(band.n))
    doping = 1-totalFilling
    if printDoping is True:
        print("total hole doping = " + "{0:.3f}".format(doping))
        print("------------------------------------------------")
    return doping

def dopingCondition(mu,p_target,bandIterable):
    print("mu = " + "{0:.3f}".format(mu))
    for band in bandIterable:
        band["mu"] = mu
    return doping(bandIterable) - p_target

def set_mu_to_doping(bandIterable, p_target, ptol=0.001):
    print("Computing mu for hole doping = " + "{0:.3f}".format(p_target))
    mu = optimize.brentq(dopingCondition, -10, 10, args=(p_target ,bandIterable), xtol=ptol)
    for band in bandIterable:
        band["mu"] = mu
