import numpy as np
from numpy import cos, sin, pi, sqrt
from scipy import optimize
from skimage import measure
from numba import jit, jitclass
import matplotlib as mpl
import matplotlib.pyplot as plt
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# Constant //////
hbar = 1  # velocity will be in units of 1 / hbar,
# this hbar is taken into accound in the constant units_move_eq

## Units ////////
meVolt = 1.602e-22  # 1 meV in Joule
Angstrom = 1e-10  # 1 A in meters


class BandStructure:
    def __init__(self, bandname="band0", a=3.74767, b=3.74767, c=13.2,
                 t=190, tp=-0.14, tpp=0.07,
                 tz=0.07, tz2=0, mu=-0.825,
                 numberOfKz=7, mesh_ds=1/20, **trash):
        self.a    = a  # in Angstrom
        self.b    = b  # in Angstrom
        self.c    = c  # in Angstrom
        self._t   = t  # meV
        self._tp  = tp  * t
        self._tpp = tpp * t
        self._tz  = tz  * t
        self._tz2 = tz2 * t
        self._mu  = mu  * t
        self.p    = None # hole doping, unknown at first
        self.n    = None # band filling (of electron), unknown at first
        self.dos  = None
        self.vf_mean  = None
        self.numberOfBZ = 1 # number of BZ we intregrate on
        self.bandname = bandname # a string to designate the band

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


    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def _get_t(self):
        return self._t
    def _set_t(self, t):
        self._tp  = self.tp  * t
        self._tpp = self.tpp * t
        self._tz  = self.tz  * t
        self._tz2 = self.tz2 * t
        self._mu  = self.mu  * t
        self._t = t
        self.erase_Fermi_surface()
    t = property(_get_t, _set_t)

    def _get_mu(self):
        return self._mu / self._t
    def _set_mu(self, mu):
        self._mu = mu * self._t
        self.erase_Fermi_surface()
    mu = property(_get_mu, _set_mu)

    def _get_tp(self):
        return self._tp / self._t
    def _set_tp(self, tp):
        self._tp = tp * self._t
        self.erase_Fermi_surface()
    tp = property(_get_tp, _set_tp)

    def _get_tpp(self):
        return self._tpp / self._t
    def _set_tpp(self, tpp):
        self._tpp = tpp * self._t
        self.erase_Fermi_surface()
    tpp = property(_get_tpp, _set_tpp)

    def _get_tz(self):
        return self._tz / self._t
    def _set_tz(self, tz):
        self._tz = tz * self._t
        self.erase_Fermi_surface()
    tz = property(_get_tz, _set_tz)

    def _get_tz2(self):
        return self._tz2 / self._t
    def _set_tz2(self, tz2):
        self._tz2 = tz2 * self._t
        self.erase_Fermi_surface()
    tz2 = property(_get_tz2, _set_tz2)

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def erase_Fermi_surface(self):
        self.kf  = None
        self.vf  = None
        self.dkf = None
        self.p    = None
        self.n    = None
        self.dos  = None
        self.vf_mean  = None
        self.numberPointsPerKz_list = []

    def bandParameters(self):
        return [self.a, self.b, self.c, self._mu, self._t, self._tp, self._tpp, self._tz, self._tz2]

    def e_3D_func(self, kx, ky, kz):
        return optimized_e_3D_func(kx, ky, kz, *self.bandParameters())

    def v_3D_func(self, kx, ky, kz):
        return optimized_v_3D_func(kx, ky, kz, *self.bandParameters())

    def dispersionMesh(self, resX=500, resY=500, resZ=10):
        kx_a = np.linspace(-pi / self.a, pi / self.a, resX)
        ky_a = np.linspace(-pi / self.b, pi / self.b, resY)
        kz_a = np.linspace(-2 * pi / self.c, 2 * pi / self.c, resZ)
        kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a, indexing='ij')
        epsilon = self.e_3D_func(kxx, kyy, kzz)
        return epsilon

    def updateFilling(self, resX=500, resY=500, resZ=10):
        epsilon = self.dispersionMesh(resX, resY, resZ)
        kVolume = epsilon.shape[0] * epsilon.shape[1] * epsilon.shape[2]
        self.n = 2 * np.sum(np.greater_equal(0, epsilon)) / kVolume / self.numberOfBZ # 2 is for the spin
        self.p = 1 - self.n
        return self.n

    def doping(self, resX=500, resY=500, resZ=10, printDoping=False):
        self.updateFilling(resX,resY,resZ)
        if printDoping==True:
            print("p=" + "{0:.3f}".format(self.p) + " :: " + self.bandname)
        return self.p

    def filling(self, resX=500, resY=500, resZ=10):
        self.updateFilling(resX,resY,resZ)
        print("n = " + "{0:.3f}".format(self.n))
        return self.n

    def dopingPerkz(self, resX=500, resY=500, resZ=10):
        epsilon = self.dispersionMesh(resX, resY, resZ)
        # Number of k in the Brillouin zone per plane
        Nz = epsilon.shape[0] * epsilon.shape[1]
        # Number of electron in the Brillouin zone per plane
        n_per_kz = 2 * np.sum(np.greater_equal(0, epsilon), axis=(0, 1)) / Nz / self.numberOfBZ # 2 is for the spin
        p_per_kz = 1 - n_per_kz
        return p_per_kz

    def diffDoping(self, mu, ptarget):
        self.mu = mu
        return self.doping() - ptarget

    def setMuToDoping(self, pTarget, ptol=0.001):
        self.mu = optimize.brentq(self.diffDoping, -10, 10, args=(pTarget,), xtol=ptol)

    def discretize_FS(self, epsilon=0, mesh_xy_rough=501, PrintEnding=False):
        """
        mesh_xy_rough: make denser rough meshgrid to interpolate after
        """

        kx_a = np.linspace(0, pi / self.a, mesh_xy_rough)
        ky_a = np.linspace(0, pi / self.b, mesh_xy_rough)
        kz_a = np.linspace(0, 2 * pi / self.c, self.numberOfKz)
        # half of FBZ, 2*pi/c because bodycentered unit cell
        dkz = 2 * pi / self.c / self.numberOfKz # integrand along z, in A^-1
        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        for j, kz in enumerate(kz_a):
            bands = self.e_3D_func(kxx, kyy, kz)
            contours = measure.find_contours(bands, epsilon)
            numberPointsPerKz = 0

            for i, contour in enumerate(contours):

                # Contour come in units proportionnal to size of meshgrid
                # one want to scale to units of kx and ky
                x = contour[:, 0] / (mesh_xy_rough - 1) * pi
                y = contour[:, 1] / (mesh_xy_rough - 1) * pi / (self.b / self.a)  # anisotropy

                ds = sqrt(np.diff(x)**2 + np.diff(y)**2)  # segment lengths
                s = np.zeros_like(x)  # arrays of zeros
                s[1:] = np.cumsum(ds)  # integrate path, s[0] = 0

                mesh_xy = int(max(np.ceil(np.max(s) / (self.mesh_ds*pi)), 4))
                # choose at least a minimum of 4 points per contour
                numberPointsPerKz += mesh_xy
                # discretize one fourth of FS, therefore need * 4

                dks = np.max(s) / (mesh_xy + 1) / self.a  # dk path

                # regular spaced path, add one
                s_int = np.linspace(0, np.max(s), mesh_xy + 1)
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
                    self.dkf = 2 * dks * dkz * np.ones_like(x_int)
                    # factor 2 because integrate only half kz.
                else:
                    kxf = np.append(kxf, x_int / self.a)
                    kyf = np.append(kyf, y_int / self.a)
                    kzf = np.append(kzf, kz * np.ones_like(x_int))
                    self.dkf = np.append(self.dkf, 2 * dks * dkz * np.ones_like(x_int))

            self.numberPointsPerKz_list.append(4 * numberPointsPerKz)

        # dim -> (n, i0) = (xyz, position on FS)
        self.kf = np.vstack([kxf, kyf, kzf])

        # Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0, :], self.kf[1, :], self.kf[2, :])
        # dim -> (i, i0) = (xyz, position on FS)
        self.vf = np.vstack([vx, vy, vz])

        ## Output message
        if PrintEnding == True:
            print("Band: " + self.bandname + ": discretized")


    def densityOfState(self):
        # Density of State
        dos = 1 / sqrt( self.vf[0,:]**2 + self.vf[1,:]**2 +self.vf[2,:]**2 )
        # dos = 1 / (hbar * |grad(E)|), here hbar is integrated in units_chambers
        self.dos = dos
        # units_vf = meVolt * Angstrom / 1.0545718e-34 # J.s
        # self.vf_mean = np.mean(sqrt( self.vf[0,:]**2+self.vf[1,:]**2+self.vf[2,:]**2 )) * units_vf
        return dos

    # def dos_epsilon(self):

    #     dos = 1 / sqrt( self.vf[0,:]**2 + self.vf[1,:]**2 +self.vf[2,:]**2 )


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

    def figMultipleFS2D(self, kz = 0, meshXY = 1001):
        """Show 2D Fermi Surface for different kz"""
        mesh_graph = meshXY
        kx = np.linspace(-4*pi / self.a, 4*pi / self.a, mesh_graph)
        ky = np.linspace(-4*pi / self.b, 4*pi / self.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

        fig, axes = plt.subplots(1, 1, figsize=(6.5, 5.6))
        fig.subplots_adjust(left=0.15, right=0.75, bottom=0.20, top=0.9)

        fig.text(0.77,0.84, r"$k_{\rm z}$ =")
        fig.text(0.88,0.84, r"0", color = "#FF0000")
        fig.text(0.88,0.78, r"$\pi/c$", color = "#00DC39")
        fig.text(0.88,0.72, r"2$\pi/c$", color = "#6577FF")

        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, pi/self.c), 0, colors = '#00DC39', linewidths = 3)
        axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 2*pi/self.c), 0, colors = '#6577FF', linewidths = 3)

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

        plt.show()
        #//////////////////////////////////////////////////////////////////////////////#









# ABOUT JUST IN TIME (JIT) COMPILATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# jitclass do not work, the best option is to call a jit otimized function from inside the class.
@jit(nopython=True, cache=True, parallel=True)
def optimized_e_3D_func(kx, ky, kz, a, b, c, mu, t, tp, tpp, tz, tz2):
    # 2D Dispersion
    e_2D = -2 * t * (cos(kx * a) + cos(ky * b))
    e_2D += -4 * tp * cos(kx * a) * cos(ky * b)
    e_2D += -2 * tpp * (cos(2 * kx * a) + cos(2 * ky * b))
    # kz dispersion
    e_z = -2 * tz * cos(kx * a / 2) * cos(ky * b / 2) * cos(kz * c / 2)
    e_z *= (cos(kx * a) - cos(ky * b))**2
    e_z += -2 * tz2 * cos(kz * c / 2)

    epsilon = e_2D + e_z - mu

    return epsilon


@jit(nopython=True, cache=True, parallel=True)
def optimized_v_3D_func(kx, ky, kz, a, b, c, mu, t, tp, tpp, tz, tz2):
    d = c / 2
    kxa = kx * a
    kyb = ky * b
    kzd = kz * d
    coskx = cos(kxa)
    cosky = cos(kyb)
    coskz = cos(kzd)
    sinkx = sin(kxa)
    sinky = sin(kyb)
    sinkz = sin(kzd)
    sin2kx = sin(2 * kxa)
    sin2ky = sin(2 * kyb)
    coskx_2 = cos(kxa / 2)
    cosky_2 = cos(kyb / 2)
    sinkx_2 = sin(kxa / 2)
    sinky_2 = sin(kyb / 2)

    # Velocity from e_2D
    d_e2D_dkx = 2 * t * a * sinkx + 4 * tp * a * sinkx * cosky + 4 * tpp * a * sin2kx
    d_e2D_dky = 2 * t * b * sinky + 4 * tp * b * coskx * sinky + 4 * tpp * b * sin2ky
    d_e2D_dkz = 0

    # Velocity from e_z
    sigma = coskx_2 * cosky_2
    diff = (coskx - cosky)
    square = (diff)**2
    d_sigma_dkx = - a / 2 * sinkx_2 * cosky_2
    d_sigma_dky = - b / 2 * coskx_2 * sinky_2
    d_square_dkx = 2 * (diff) * (-a * sinkx)
    d_square_dky = 2 * (diff) * (+b * sinky)

    d_ez_dkx = - 2 * tz * d_sigma_dkx * square * coskz - 2 * tz * sigma * d_square_dkx * coskz
    d_ez_dky = - 2 * tz * d_sigma_dky * square * coskz - 2 * tz * sigma * d_square_dky * coskz
    d_ez_dkz = - 2 * tz * sigma * square * (-d * sinkz) - 2 * tz2 * (-d) * sinkz

    vx = (d_e2D_dkx + d_ez_dkx) / hbar
    vy = (d_e2D_dky + d_ez_dky) / hbar
    vz = (d_e2D_dkz + d_ez_dkz) / hbar

    return vx, vy, vz


@jit(nopython=True, cache=True, parallel=True)
def rotation(x, y, angle):
    xp = cos(angle) * x + sin(angle) * y
    yp = -sin(angle) * x + cos(angle) * y
    return xp, yp









## Antiferromagnetic Reconstruction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
class Pocket(BandStructure):
    def __init__(self, M=0.2, electronPocket=False, **kwargs):
        super().__init__(**kwargs)
        self._M = M*self.t
        self.electronPocket = electronPocket
        self.numberOfBZ = 2  # number of BZ we intregrate on as we still work on the unreconstructed FBZ

    def _get_M(self):
        return self._M / self._t
    def _set_M(self, M):
        self._M = M * self._t
    M = property(_get_M, _set_M)

    def e_3D_func(self, kx, ky, kz):
        return optimizedAFfuncs(kx, ky, kz, self._M, *self.bandParameters(), self.electronPocket)[0]

    def v_3D_func(self, kx, ky, kz):
        return optimizedAFfuncs(kx, ky, kz, self._M, *self.bandParameters(), self.electronPocket)[1:]



## These definition are only valid for (Qx,Qy)=(pi,pi) without coherence of the AF in z
@jit(nopython=True, cache=True, parallel=True)
def optimizedAFfuncs(kx, ky, kz, M, a, b, c, mu, t, tp, tpp, tz, tz2, electronPocket=False):
    d = c / 2
    kxa = kx * a
    kyb = ky * b
    kzd = kz * d
    coskx = cos(kxa)
    cosky = cos(kyb)
    coskz = cos(kzd)
    sinkx = sin(kxa)
    sinky = sin(kyb)
    sinkz = sin(kzd)
    cos2kx = cos(2 * kxa)
    cos2ky = cos(2 * kyb)
    sin2kx = sin(2 * kxa)
    sin2ky = sin(2 * kyb)
    coskx_2 = cos(kxa / 2)
    cosky_2 = cos(kyb / 2)
    sinkx_2 = sin(kxa / 2)
    sinky_2 = sin(kyb / 2)

    # Decomposition: epsilon(k) = zeta(k) + xi(k)  to take advantage of zeta(k+Q) = zeta(k)
    zeta_k      = -4.*tp*coskx*cosky - 2.*tpp*(cos2kx + cos2ky) - mu
    dzeta_k_dkx =  4.*a*tp*sinkx*cosky + 4.*a*tpp*sin2kx
    dzeta_k_dky =  4.*b*tp*coskx*sinky + 4.*b*tpp*sin2ky
    xi_k      =   -2.*t*(coskx + cosky)
    dxi_k_dkx =    2.*a*t*sinkx
    dxi_k_dky =    2.*b*t*sinky

    epsilon_k      = xi_k           + zeta_k
    depsilon_k_dkx = dxi_k_dkx      + dzeta_k_dkx
    depsilon_k_dky = dxi_k_dky      + dzeta_k_dky

    epz_k    = -2*tz*coskz* (coskx-cosky)**2 *coskx_2*cosky_2 -2 * tz2 * cos(kz * d)

    # kz dispersion and its derivatives (won't be affected by Q)
    # Full simplified with mathematica
    # depz_dkx = tz*cosky_2*(-4-5*coskx+cosky)*(-coskx+cosky)*coskz*a*sinkx_2
    # depz_dky = tz*coskx_2*(-4+coskx-5*cosky)*(coskx-cosky)*coskz*b*sinky_2
    # depz_dkz = tz*coskx_2*cosky_2*(coskx-cosky)*(coskx-cosky)*c*sinkz  +tz2*d*sinkz

    sigma = coskx_2 * cosky_2
    diff = (coskx - cosky)
    square = (diff)**2
    d_sigma_dkx = - a/2 * sinkx_2 * cosky_2
    d_sigma_dky = - b/2 * coskx_2 * sinky_2
    d_square_dkx = 2 * (diff) * (-a * sinkx)
    d_square_dky = 2 * (diff) * (+b * sinky)
    depz_dkx = - 2 * tz * d_sigma_dkx * square * coskz - 2 * tz * sigma * d_square_dkx * coskz
    depz_dky = - 2 * tz * d_sigma_dky * square * coskz - 2 * tz * sigma * d_square_dky * coskz
    depz_dkz = - 2 * tz * sigma * square * (-d * sinkz) - 2 * tz2 * (-d) * sinkz

    # AF EIGENVALUES
    #epsilon(k+Q) and its derivatives
    epsilon_kQ           = -xi_k           + zeta_k
    depsilon_kQ_dkx      = -dxi_k_dkx      + dzeta_k_dkx
    depsilon_kQ_dky      = -dxi_k_dky      + dzeta_k_dky

    #precalculate sum, diff and radical:
    Sk          = 0.5*(  epsilon_k         +   epsilon_kQ)
    dSk_dkx     = 0.5*( depsilon_k_dkx     +  depsilon_kQ_dkx)
    dSk_dky     = 0.5*( depsilon_k_dky     +  depsilon_kQ_dky)
    Dk          = 0.5*(  epsilon_k         -   epsilon_kQ)
    dDk_dkx     = 0.5*( depsilon_k_dkx     -  depsilon_kQ_dkx)
    dDk_dky     = 0.5*( depsilon_k_dky     -  depsilon_kQ_dky)

    if M<=0.00001:
        Rk = Dk
        dRk_dkx = dDk_dkx
        dRk_dky = dDk_dky
    else:
        Rk          = sqrt( Dk*Dk + M*M )
        dRk_dkx     = Dk*dDk_dkx/Rk
        dRk_dky     = Dk*dDk_dky/Rk

    #finally calculate the eigenvalues and their derivatives (vertices):
    if electronPocket:
        Ek          =   Sk         +   Rk
        dEk_dkx     =  dSk_dkx     +  dRk_dkx
        dEk_dky     =  dSk_dky     +  dRk_dky
    else: #holePocket
        Ek          =   Sk         -   Rk
        dEk_dkx     =  dSk_dkx     -  dRk_dkx
        dEk_dky     =  dSk_dky     -  dRk_dky

    return Ek+epz_k, (dEk_dkx+depz_dkx)/hbar, (dEk_dky+depz_dky)/hbar, depz_dkz/hbar







## Functions to compute the doping of a two bands system and more >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
def doping(bandIterable, printDoping=False):
    totalFilling=0
    if printDoping == True:
        print("------------------------------------------------")
    for band in bandIterable:
        band.updateFilling()
        totalFilling += band.n
        if printDoping == True:
            print(band.bandname + ": band filling = " + "{0:.3f}".format(band.n))
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
