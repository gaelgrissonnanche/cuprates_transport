import numpy as np
from numpy import cos, sin, pi, sqrt
from scipy import optimize
from skimage import measure
from numba import jit
import matplotlib as mpl
import matplotlib.pyplot as plt
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# Constant //////
hbar = 1  # velocity will be in units of 1 / hbar,
# this hbar is taken into accound in the constant units_move_eq

class BandStructure:
    def __init__(self, a=3.74, b=3.74, c=13.3,
                 t=190, tp=-0.14, tpp=0.07, tz=0.07, tz2=0.00, mu=-0.825,
                 numberOfKz=7, mesh_ds=pi/20):
        self.a   = a  # in Angstrom
        self.b   = b  # in Angstrom
        self.c   = c  # in Angstrom
        self.t   = t  # meV
        self.tp  = tp  * self.t
        self.tpp = tpp * self.t
        self.tz  = tz  * self.t
        self.tz2 = tz2 * self.t
        self.mu  = mu  * self.t
        self.p   = None # hole doping, unknown at first

        ## Discretization
        self.mesh_ds    = mesh_ds  # length resolution in FBZ in units of Pi
        if numberOfKz % 2 == 0:  # make sure it is an odd number
            numberOfKz += 1
        self.numberOfKz = numberOfKz  # between 0 and 2*pi / c

        ## Fermi surface arrays
        self.kf  = None
        self.vf  = None
        self.dkf = None

        ## Save number of points in each kz plane
        self.numberPointsPerKz_list = []

        ## Make initial discretization of the Fermi surface within constructor
        self.discretize_FS()

        ## Compute the doping
        self.doping()
        print("p = " + "{0:.3f}".format(self.p))

        ## Compute Density of State
        self.dos = self.densityOfState()

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def bandParameters(self):
        return [self.a, self.b, self.c, self.mu, self.t, self.tp, self.tpp, self.tz, self.tz2]


    def e_3D_func(self, kx, ky, kz):
        return optimized_e_3D_func(kx, ky, kz, self.a, self.b, self.c, self.mu, self.t, self.tp, self.tpp, self.tz, self.tz2)

    def v_3D_func(self, kx, ky, kz):
        return optimized_v_3D_func(kx, ky, kz, self.a, self.b, self.c, self.t, self.tp, self.tpp, self.tz, self.tz2)

    def dispersionMesh(self, resX=500, resY=500, resZ=10):
        kx_a = np.linspace(-pi / self.a, pi / self.a, resX)
        ky_a = np.linspace(-pi / self.b, pi / self.b, resY)
        kz_a = np.linspace(-2 * pi / self.c, 2 * pi / self.c, resZ)
        kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a, indexing='ij')

        disp = self.e_3D_func(kxx, kyy, kzz)
        return disp

    def doping(self, resX=500, resY=500, resZ=10):
        E = self.dispersionMesh(resX, resY, resZ)

        # Number of k in the total Brillouin Zone
        N = E.shape[0] * E.shape[1] * E.shape[2]
        # Number of electron in the total Brillouin Zone
        n = 2 / N * np.sum(np.greater_equal(0, E))
                # number of quasiparticles below mu, 2 is for the spin
        # Number of holes
        self.p = 1 - n

        return self.p

    def dopingPerkz(self, resX=500, resY=500, resZ=10):
        E = self.dispersionMesh(resX, resY, resZ)

        # Number of k in the Brillouin zone per plane
        Nz = E.shape[0] * E.shape[1]
        # Number of electron in the Brillouin zone per plane
        n_per_kz = 2 / Nz * np.sum(np.greater_equal(0, E), axis=(0, 1))
        p_per_kz = 1 - n_per_kz

        return p_per_kz

    def dopingCondition(self, mu, ptarget):
        self.mu = mu
        print(self.doping())
        return self.doping() - ptarget


    def setMuToDoping(self, pTarget, muStart=-8.0, xtol=0.001):
        solObject = optimize.root(self.dopingCondition, np.array(
            [muStart]), args=(pTarget,), options={'xtol': xtol})
        self.mu = solObject.x[0]


    def discretize_FS(self):
        mesh_xy_rough = 501  # make denser rough meshgrid to interpolate after
        kx_a = np.linspace(0, pi / self.a, mesh_xy_rough)
        ky_a = np.linspace(0, pi / self.b, mesh_xy_rough)
        kz_a = np.linspace(0, 2 * pi / self.c, self.numberOfKz)
               # half of FBZ, 2*pi/c because bodycentered unit cell
        dkz = kz_a[1] - kz_a[0] # integrand along z, in A^-1
        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        for j, kz in enumerate(kz_a):
            bands = self.e_3D_func(kxx, kyy, kz)
            contours = measure.find_contours(bands, 0)
            numberPointsPerKz = 0

            for i, contour in enumerate(contours):

                # Contour come in units proportionnal to size of meshgrid
                # one want to scale to units of kx and ky
                x = contour[:, 0] / (mesh_xy_rough - 1) * pi
                y = contour[:, 1] / (mesh_xy_rough - 1) * pi / (self.b / self.a)  # anisotropy

                ds = sqrt(np.diff(x)**2 + np.diff(y)**2)  # segment lengths
                s = np.zeros_like(x)  # arrays of zeros
                s[1:] = np.cumsum(ds)  # integrate path, s[0] = 0

                mesh_xy = int(max(np.ceil(s.max() / self.mesh_ds), 4))
                          # choose at least a minimum of 4 points per contour
                numberPointsPerKz += mesh_xy
                          # discretize one fourth of FS, therefore need * 4

                dpath = s.max() / (mesh_xy + 1)  # weight to ponderate self.dkf

                # regular spaced path, add one
                s_int = np.linspace(0, s.max(), mesh_xy + 1)
                # interpolate and remove the last point (not to repeat)
                x_int = np.interp(s_int, s, x)[:-1]
                y_int = np.interp(s_int, s, y)[:-1]

                # Rotate the contour to get the entire Fermi surface
                # ### WARNING NOT ROBUST IN THE CASE OF C4 SYMMETRY BREAKING
                x_dump = x_int
                y_dump = y_int
                for angle in [pi / 2, pi, 3 * pi / 2]:
                    x_int_p, y_int_p = rotation(x_int, y_int, angle)
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
                    self.dkf = 2 * dpath * dkz * np.ones_like(x_int)
                                        # factor because integrate only half kz.
                else:
                    kxf = np.append(kxf, x_int / self.a)
                    # self.a (and not b) because anisotropy is taken into account earlier
                    kyf = np.append(kyf, y_int / self.a)
                    kzf = np.append(kzf, kz * np.ones_like(x_int))
                    self.dkf = np.append(self.dkf, 2 * dpath * dkz * np.ones_like(x_int))

            self.numberPointsPerKz_list.append(4 * numberPointsPerKz)

        # dim -> (n, i0) = (xyz, position on FS)
        self.kf = np.vstack([kxf, kyf, kzf])

        # Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0, :], self.kf[1, :], self.kf[2, :])
        # dim -> (i, i0) = (xyz, position on FS)
        self.vf = np.vstack([vx, vy, vz])

    def densityOfState(self):
        # Density of State
        dos = 1 / sqrt( self.vf[0,:]**2 + self.vf[1,:]**2 +self.vf[2,:]**2 )
                # dos = 1 / (hbar * |grad(E)|), here hbar is integrated in units_chambers
        return dos

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

        for tick in axes.xaxis.get_major_ticks():
            tick.set_pad(7)
        for tick in axes.yaxis.get_major_ticks():
            tick.set_pad(8)

        fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
        line = axes.plot(self.kf[0,:self.numberPointsPerKz_list[0]] * self.a,
                         self.kf[1,:self.numberPointsPerKz_list[0]] * self.b)
        plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
        axes.quiver(self.kf[0,:self.numberPointsPerKz_list[0]] * self.a,
                    self.kf[1,:self.numberPointsPerKz_list[0]] * self.b,
                    self.vf[0,:self.numberPointsPerKz_list[0]],
                    self.vf[1,:self.numberPointsPerKz_list[0]],
                    color = 'k')

        axes.set_xlim(-pi, pi)
        axes.set_ylim(-pi, pi)
        axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
        axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

        axes.set_xticks([-pi, 0., pi])
        axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axes.set_yticks([-pi, 0., pi])
        axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

        plt.show()
        #//////////////////////////////////////////////////////////////////////#

    def figDiscretizeFS3D(self):
        """Show Discretized 3D Fermi Surface """
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.kf[0], self.kf[1], self.kf[2], color='k', marker='.')
        plt.show()

    def figMultipleFS2D(self, kz = 0, meshXY = 1001):
        """Show 2D Fermi Surface for different kz"""
        mesh_graph = meshXY
        kx = np.linspace(-pi / self.a, pi / self.a, mesh_graph)
        ky = np.linspace(-pi / self.b, pi / self.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

        fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6)) # (1,1) means one plot, and figsize is w x h in inch of figure
        fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91) # adjust the box of axes regarding the figure size

        for tick in axes.xaxis.get_major_ticks():
            tick.set_pad(7)
        for tick in axes.yaxis.get_major_ticks():
            tick.set_pad(8)

        fig.text(0.27,0.86, r"$k_{\rm z}$ =", fontsize = 14)
        fig.text(0.34,0.86, r"0", fontsize = 14, color = "#FF0000")
        fig.text(0.34,0.83, r"$\pi/c$", fontsize = 14, color = "#00DC39")
        fig.text(0.34,0.80, r"2$\pi/c$", fontsize = 14, color = "#6577FF")

        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, pi/self.c), 0, colors = '#00DC39', linewidths = 3)
        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 2*pi/self.c), 0, colors = '#6577FF', linewidths = 3)

        axes.set_xlim(-pi, pi)
        axes.set_ylim(-pi, pi)
        axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
        axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

        axes.set_xticks([-pi, 0., pi])
        axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        axes.set_yticks([-pi, 0., pi])
        axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

        plt.show()
        #//////////////////////////////////////////////////////////////////////////////#



# ABOUT JUST IN TIME (JIT) COMPILATION
# jitclass do not work, the best option is to call a jit otimized function from inside the class.
@jit(nopython=True, cache=True)
def optimized_e_3D_func(kx, ky, kz, a, b, c, mu, t, tp, tpp, tz, tz2):
    # 2D Dispersion
    e_2D = -2 * t * (cos(kx * a) + cos(ky * b))
    e_2D += -4 * tp * cos(kx * a) * cos(ky * b)
    e_2D += -2 * tpp * (cos(2 * kx * a) + cos(2 * ky * b))
    e_2D += -mu
    # Kz dispersion
    d = c / 2.
    e_z = -2 * tz * cos(kz * d)
    e_z *= cos(kx * a / 2) * cos(ky * b / 2)
    e_z *= (cos(kx * a) - cos(ky * b))**2
    e_z += -2 * tz2 * cos(kz * d) # mind the '+='

    return e_2D + e_z


@jit(nopython=True, cache=True)
def optimized_v_3D_func(kx, ky, kz, a, b, c, t, tp, tpp, tz, tz2):
    d = c / 2
    kxa = kx * a
    kxb = ky * b
    kzd = kz * d
    coskx = cos(kxa)
    cosky = cos(kxb)
    coskz = cos(kzd)
    sinkx = sin(kxa)
    sinky = sin(kxb)
    sinkz = sin(kzd)
    sin2kx = sin(2 * kx * a)
    sin2ky = sin(2 * ky * b)
    coskx_2 = cos(kxa / 2)
    cosky_2 = cos(kxb / 2)
    sinkx_2 = sin(kxa / 2)
    sinky_2 = sin(kxb / 2)

    # Velocity from e_2D
    d_e2D_dkx = 2 * t * a * sinkx + 4 * tp * \
        a * sinkx * cosky + 4 * tpp * a * sin2kx
    d_e2D_dky = 2 * t * b * sinky + 4 * tp * \
        b * coskx * sinky + 4 * tpp * b * sin2ky
    d_e2D_dkz = 0

    # Velocity from e_z
    sigma = coskx_2 * cosky_2
    diff = (coskx - cosky)
    square = (diff)**2
    d_sigma_dkx = - a / 2 * sinkx_2 * cosky_2
    d_sigma_dky = - b / 2 * coskx_2 * sinky_2
    d_square_dkx = 2 * (diff) * (-a * sinkx)
    d_square_dky = 2 * (diff) * (+b * sinky)

    d_ez_dkx = - 2 * tz * d_sigma_dkx * square * \
        coskz - 2 * tz * sigma * d_square_dkx * coskz
    d_ez_dky = - 2 * tz * d_sigma_dky * square * \
        coskz - 2 * tz * sigma * d_square_dky * coskz
    d_ez_dkz = - 2 * tz * sigma * square * \
        (-d * sinkz) + 2 * tz2 * (-d) * sinkz

    vx = (d_e2D_dkx + d_ez_dkx) / hbar
    vy = (d_e2D_dky + d_ez_dky) / hbar
    vz = (d_e2D_dkz + d_ez_dkz) / hbar

    return vx, vy, vz


def rotation(x, y, angle):
    xp = cos(angle) * x + sin(angle) * y
    yp = -sin(angle) * x + cos(angle) * y
    return xp, yp
