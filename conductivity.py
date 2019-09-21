import numpy as np
from numpy import cos, sin, pi, exp, sqrt, arctan2
from scipy.integrate import odeint
from skimage import measure
from numba import jit, prange
import matplotlib as mpl
import matplotlib.pyplot as plt
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Constant //////
hbar = 1.05e-34 # m2 kg / s
e = 1.6e-19 # C

## Units ////////
meVolt = 1.602e-22 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 1e-12 # 1 ps in seconds

## This coefficient takes into accound all units and constant to prefactor the movement equation
units_move_eq =  e * Angstrom**2 * picosecond * meVolt / hbar**2

## This coefficient takes into accound all units and constant to prefactor Chambers formula
units_chambers = 2 * e**2 / (2*pi)**3 * meVolt * picosecond / Angstrom / hbar**2


class Conductivity:
    def __init__(self, bandObject, Bamp, Bphi=0, Btheta=0,
                 gamma_0=15, gamma_dos_max=0, gamma_k=0, power=2, factor_arcs=1):

        # Band object
        self.bandObject = bandObject ## WARNING do not modify within this object

        # Magnetic field in degrees
        self._Bamp   = Bamp
        self._Btheta = Btheta
        self._Bphi   = Bphi
        self._B_vector = self.BFunc() # np array fo Bx,By,Bz

        # Scattering rate
        self.gamma_0 = gamma_0 # in THz
        self.gamma_dos_max = gamma_dos_max # in THz
        self.gamma_k = gamma_k # in THz
        self.power   = int(power)
        if self.power % 2 == 1:
            self.power += 1
        self.factor_arcs = factor_arcs # factor * gamma_0 outsite AF FBZ
        self.gamma_tot_max = 1 / self.tauTotMinFunc() # in THz
        self.gamma_tot_min = 1 / self.tauTotMaxFunc() # in THz

        # Time parameters
        self.tmax = 8 * self.tauTotMaxFunc()  # in picoseconds
        self._Ntime = 500 # number of steps in time
        self.dt = self.tmax / self.Ntime
        self.t = np.arange(0, self.tmax, self.dt)
        self.dt_array = np.append(0, self.dt * np.ones_like(self.t))[:-1] # integrand for tau_function

        # Time-dependent kf, vf
        self.kft = np.empty(1)
        self.vft = np.empty(1)
        self.t_over_tau = np.empty(1) # array[i0, i_t] with i0 index of the initial index
        # i_t index of the time from the starting position

        # Conductivity Tensor: x, y, z = 0, 1, 2
        self.sigma = np.empty((3,3), dtype= np.float64)



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
        self._B_vector = self.BFunc()
    Bamp = property(_get_Bamp, _set_Bamp)

    def _get_Bphi(self):
        return self._Bphi
    def _set_Bphi(self, Bphi):
        self._Bphi = Bphi
        self._B_vector = self.BFunc()
    Bphi = property(_get_Bphi, _set_Bphi)

    def _get_Btheta(self):
        return self._Btheta
    def _set_Btheta(self, Btheta):
        self._Btheta = Btheta
        self._B_vector = self.BFunc()
    Btheta = property(_get_Btheta, _set_Btheta)

    def _get_Ntime(self):
        return self._Ntime
    def _set_Ntime(self, Ntime):
        self._Ntime = Ntime
        self.dt = self.tmax / self._Ntime
        self.t = np.arange(0, self.tmax, self.dt) # integrand for tau_function
        self.dt_array = np.append(0, self.dt * np.ones_like(self.t))[:-1]
    Ntime = property(_get_Ntime, _set_Ntime)



    ## Special Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def __eq__(self, other):
        return (
                self._Bamp   == other._Bamp
            and self._Btheta == other._Btheta
            and self._Bphi   == other._Bphi
            and np.all(self.sigma == other.sigma)
        )

    def __ne__(self, other):
        return not self == other




    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def BFunc(self):
        B = self._Bamp * \
            np.array([sin(self._Btheta*pi/180) * cos(self._Bphi*pi/180),
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

    def solveMovementFunc(self):
        len_t = self.t.shape[0]
        len_kf = self.bandObject.kf.shape[1]
        # Flatten to get all the initial kf solved at the same time
        self.bandObject.kf.shape = (3 * len_kf,)
        # Sovle differential equation
        self.kft = odeint(self.diffEqFunc, self.bandObject.kf, self.t, rtol = 1e-4, atol = 1e-4).transpose()
        # Reshape arrays
        self.bandObject.kf.shape = (3, len_kf)
        self.kft.shape = (3, len_kf, len_t)
        # Velocity function of time
        self.vft = np.empty_like(self.kft, dtype = np.float64)
        self.vft[0, :, :], self.vft[1, :, :], self.vft[2, :, :] = self.bandObject.v_3D_func(self.kft[0, :, :], self.kft[1, :, :], self.kft[2, :, :])
        # Compute right away the t_over_tau array
        self.tOverTauFunc()

    def diffEqFunc(self, k, t):
        len_k = int(k.shape[0]/3)
        k.shape = (3, len_k) # reshape the flatten k
        vx, vy, vz =  self.bandObject.v_3D_func(k[0,:], k[1,:], k[2,:])
        dkdt = ( - units_move_eq ) * self.crossProductVectorized(vx, vy, vz)
        dkdt.shape = (3*len_k,) # flatten k again
        return dkdt

    # def factor_arcs_Func(self):
    #     # line ky = kx + pi
    #     d1 = self.kft[1, :, :] * self.bandObject.b - self.kft[0, :, :] * self.bandObject.a - pi  # line ky = kx + pi
    #     d2 = self.kft[1, :, :] * self.bandObject.b - self.kft[0, :, :] * self.bandObject.a + pi  # line ky = kx - pi
    #     d3 = self.kft[1, :, :] * self.bandObject.b + self.kft[0, :, :] * self.bandObject.a - pi  # line ky = -kx + pi
    #     d4 = self.kft[1, :, :] * self.bandObject.b + self.kft[0, :, :] * self.bandObject.a + pi  # line ky = -kx - pi

    #     is_in_FBZ_AF = np.logical_and((d1 <= 0)*(d2 >= 0), (d3 <= 0)*(d4 >= 0))
    #     is_out_FBZ_AF = np.logical_not(is_in_FBZ_AF)
    #     factor_out_of_FBZ_AF = np.ones_like(self.kft[0, :, :])
    #     factor_out_of_FBZ_AF[is_out_FBZ_AF] = self.factor_arcs
    #     return factor_out_of_FBZ_AF

    def gamma_DOS_Func(self, vx, vy, vz):
        dos = 1 / sqrt( vx**2 + vy**2 + vz**2 )
        dos_max = np.max(self.bandObject.dos)  # value to normalize the DOS to a quantity without units
        return self.gamma_dos_max * dos / dos_max

    def gamma_k_Func(self, kx, ky):
        phi = arctan2(ky, kx)
        return self.gamma_k * cos(2*phi)**self.power

    def tOverTauFunc(self):
        # Integral from 0 to t of dt' / tau( k(t') ) or dt' * gamma( k(t') )
        self.t_over_tau = np.cumsum( self.dt_array / (
                                     self.tauTotFunc(self.kft[0, :, :],
                                                     self.kft[1, :, :],
                                                     self.vft[0, :, :],
                                                     self.vft[1, :, :],
                                                     self.vft[2, :, :]
                                                     )
                                                      )
                         , axis = 1)

    def tauTotFunc(self, kx, ky, vx, vy, vz):
        tauTot = 1 / (self.gamma_0 + # * self.factor_arcs_Func() +
                      self.gamma_DOS_Func(vx, vy, vz) +
                      self.gamma_k_Func(kx, ky))
        return tauTot

    def tauTotMaxFunc(self):
        # Compute the tau_max (the longest time between two collisions)
        # to better integrate from 0 --> 8 * 1 / gamma_min (instead of infinity)
        kf = self.bandObject.kf
        vf = self.bandObject.vf
        tauTotMax = np.max(self.tauTotFunc(kf[0, :], kf[1, :],
                                           vf[0, :], vf[1, :], vf[2, :]))
        return tauTotMax

    def tauTotMinFunc(self):
        kf = self.bandObject.kf
        vf = self.bandObject.vf
        tauTotMin = np.min(self.tauTotFunc(kf[0, :], kf[1, :],
                                           vf[0, :], vf[1, :], vf[2, :]))
        return tauTotMin

    def solveMovementForPoint(self, kpoint):
        len_t = self.t.shape[0]
        kt = odeint(self.diffEqFunc, kpoint, self.t, rtol = 1e-4, atol = 1e-4).transpose() # solve differential equation
        kt = np.reshape(kt, (3, 1, len_t))
        return kt

    def VelocitiesProduct(self, i, j):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0: vif = vxf """
        # Velocity components
        vif  = self.bandObject.vf[i,:]
        vjft = self.vft[j,:,:]
        # Integral over t
        vj_sum_over_t = np.sum(vjft * exp(-self.t_over_tau) * self.dt, axis=1)
        # Product of velocities
        v_product = vif * vj_sum_over_t
        return v_product

    def chambersFunc(self, i = 2, j = 2):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0 and j = 1 : sigma[i,j] = sigma_xy """
        # if AF reconstructed, only 1 particule per FBZ instead of 2 (spins)
        self.sigma[i, j] = units_chambers / self.bandObject.numberOfBZ * \
                           np.sum( self.bandObject.dos *
                                   self.bandObject.dkf *
                                   self.VelocitiesProduct(i=i, j=j) )



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

    # def figLifeTime(self, mesh_phi = 1000):
    #     fig, axes = plt.subplots(1, 1, figsize=(6.5, 5.6))
    #     fig.subplots_adjust(left=0.10, right=0.70, bottom=0.20, top=0.9)

    #     phi = np.linspace(0, 2*pi, 1000)
    #     ## tau_0
    #     tau_0_x = self.tau_0 * cos(phi)
    #     tau_0_y = self.tau_0 * sin(phi)
    #     line = axes.plot(tau_0_x / self.tau_0, tau_0_y / self.tau_0,
    #                          clip_on=False, label=r"$\tau_{\rm 0}$", zorder=10)
    #     plt.setp(line, ls="--", c='#000000', lw=3, marker="",
    #              mfc='#000000', ms=5, mec="#7E2320", mew=0)
    #     ## tau_k
    #     tau_k_x = 1 / (self.gamma_0 + self.gamma_k * (sin(phi) **
    #                                         2 - cos(phi)**2)**self.power) * cos(phi)
    #     tau_k_y = 1 / (self.gamma_0 + self.gamma_k * (sin(phi) **
    #                                         2 - cos(phi)**2)**self.power) * sin(phi)
    #     line = axes.plot(tau_k_x / self.tau_0, tau_k_y / self.tau_0,
    #                          clip_on=False, zorder=20, label=r"$\tau_{\rm tot}$")
    #     plt.setp(line, ls="-", c='#00FF9C', lw=3, marker="",
    #              mfc='#000000', ms=5, mec="#7E2320", mew=0)
    #     ## tau_k_min
    #     phi_min = 3 * pi / 2
    #     tau_k_x_min = 1 / \
    #         (self.gamma_0 + self.gamma_k * (sin(phi_min)**2 -
    #                               cos(phi_min)**2)**self.power) * cos(phi_min)
    #     tau_k_y_min = 1 / \
    #         (self.gamma_0 + self.gamma_k * (sin(phi_min)**2 -
    #                               cos(phi_min)**2)**self.power) * sin(phi_min)
    #     line = axes.plot(tau_k_x_min / self.tau_0, tau_k_y_min / self.tau_0,
    #                          clip_on=False, label=r"$\tau_{\rm min}$", zorder=25)
    #     plt.setp(line, ls="", c='#1CB7FF', lw=3, marker="o",
    #              mfc='#1CB7FF', ms=8, mec="#1CB7FF", mew=2)
    #     ## tau_k_max
    #     phi_max = 5 * pi / 4
    #     tau_k_x_max = 1 / \
    #         (self.gamma_0 + self.gamma_k * (sin(phi_max)**2 -
    #                               cos(phi_max)**2)**self.power) * cos(phi_max)
    #     tau_k_y_max = 1 / \
    #         (self.gamma_0 + self.gamma_k * (sin(phi_max)**2 -
    #                               cos(phi_max)**2)**self.power) * sin(phi_max)
    #     line = axes.plot(tau_k_x_max / self.tau_0, tau_k_y_max / self.tau_0,
    #                          clip_on=False, label=r"$\tau_{\rm max}$", zorder=25)
    #     plt.setp(line, ls="", c='#FF8181', lw=3, marker="o",
    #              mfc='#FF8181', ms=8, mec="#FF8181", mew=2)

    #     fraction = np.abs(np.round(self.tau_0 / tau_k_y_min, 2))
    #     fig.text(0.74, 0.21, r"$\tau_{\rm max}$/$\tau_{\rm min}$" +
    #              "\n" + r"= {0:.1f}".format(fraction))

    #     axes.set_xlim(-1, 1)
    #     axes.set_ylim(-1, 1)
    #     axes.set_xticks([])
    #     axes.set_yticks([])

    #     plt.legend(bbox_to_anchor=[1.51, 1.01], loc=1,
    #                frameon=False,
    #                numpoints=1, markerscale=1, handletextpad=0.2)

    #     plt.show()

    def figArcs(self, index_kf = 0, meshXY = 1001):
        fig, axes = plt.subplots(1, 1, figsize=(5.6, 5.6))
        fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91)

        fig.text(0.39,0.86, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)
        fig.text(0.85, 0.86, r"$\Gamma_{\rm 0}$", ha="right", fontsize=16, color = "#1CB7FF")
        fig.text(0.85, 0.81, str(self.factor_arcs) + r" $\Gamma_{\rm 0}$", ha="right", fontsize=16, color="#FF8181")

        kx = np.linspace(-pi, pi, meshXY)
        ky = np.linspace(-pi, pi, meshXY)

        # line ky = kx + pi
        d1 = ky - kx - pi  # line ky = kx + pi
        d2 = ky - kx + pi  # line ky = kx - pi
        d3 = ky + kx - pi  # line ky = -kx + pi
        d4 = ky + kx + pi  # line ky = -kx - pi

        # Draw FBZ AF
        line = axes.plot(kx, d1-ky)
        plt.setp(line, ls ="--", c = 'k', lw = 1, marker = "", mfc = 'k', ms = 5, mec = "k", mew= 0, zorder = -1)  # end point
        line = axes.plot(kx, d2-ky)
        plt.setp(line, ls ="--", c = 'k', lw = 1, marker = "", mfc = 'k', ms = 5, mec = "k", mew= 0, zorder = -1)  # end point
        line = axes.plot(kx, d3-ky)
        plt.setp(line, ls ="--", c = 'k', lw = 1, marker = "", mfc = 'k', ms = 5, mec = "k", mew= 0, zorder = -1)  # end point
        line = axes.plot(kx, d4-ky)
        plt.setp(line, ls ="--", c = 'k', lw = 1, marker = "", mfc = 'k', ms = 5, mec = "k", mew= 0, zorder = -1)  # end point

        # Draw FS
        a = self.bandObject.a
        b = self.bandObject.b
        kxx, kyy = np.meshgrid(kx/a, ky/b, indexing='ij')
        bands = self.bandObject.e_3D_func(kxx, kyy, 0)
        contours = measure.find_contours(bands, 0)

        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx_f = (contour[:, 0] / (meshXY - 1) -0.5) * 2*pi
            ky_f = (contour[:, 1] / (meshXY - 1) -0.5) * 2*pi / (b/a)

            d1 = ky_f - kx_f - pi  # line ky = kx + pi
            d2 = ky_f - kx_f + pi  # line ky = kx - pi
            d3 = ky_f + kx_f - pi  # line ky = -kx + pi
            d4 = ky_f + kx_f + pi  # line ky = -kx - pi

            is_in_FBZ_AF = np.logical_and((d1 <= 0)*(d2 >= 0), (d3 <= 0)*(d4 >= 0))
            kx_arcs = kx_f[is_in_FBZ_AF]
            ky_arcs = ky_f[is_in_FBZ_AF]
            kx_out_arcs = kx_f[np.logical_not(is_in_FBZ_AF)]
            ky_out_arcs = ky_f[np.logical_not(is_in_FBZ_AF)]

            line = axes.plot(kx_arcs, ky_arcs)
            plt.setp(line, ls="", c='#1CB7FF', lw=2, marker="o", mfc='#1CB7FF',
                 ms=2, mec='#1CB7FF', mew=0)
            line = axes.plot(kx_out_arcs, ky_out_arcs)
            plt.setp(line, ls="", c='#FF8181', lw=2, marker="o", mfc='#FF8181',
                 ms=2, mec='#FF8181', mew=0)

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


    def figOnekft(self, index_kf = 0, meshXY = 1001):
        mesh_graph = meshXY
        kx = np.linspace(-pi / self.bandObject.a, pi / self.bandObject.a, mesh_graph)
        ky = np.linspace(-pi / self.bandObject.b, pi / self.bandObject.b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

        fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6))
        fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91)

        fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

        line = axes.contour(kxx*self.bandObject.a, kyy*self.bandObject.b, self.bandObject.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
        line = axes.plot(self.kft[0, index_kf,:]*self.bandObject.a, self.kft[1, index_kf,:]*self.bandObject.b)
        plt.setp(line, ls ="-", c = 'b', lw = 1, marker = "", mfc = 'b', ms = 5, mec = "#7E2320", mew= 0) # trajectory
        line = axes.plot(self.bandObject.kf[0, index_kf]*self.bandObject.a, self.bandObject.kf[1, index_kf]*self.bandObject.b)
        plt.setp(line, ls ="", c = 'b', lw = 3, marker = "o", mfc = 'w', ms = 4.5, mec = "b", mew= 1.5)  # starting point
        line = axes.plot(self.kft[0, index_kf, -1]*self.bandObject.a, self.kft[1, index_kf, -1]*self.bandObject.b)
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

        line = axes.plot(self.t, self.vft[2, index_kf,:])
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

        line = axes.plot(self.t, np.cumsum(self.vft[2, index_kf, :] * exp(-self.t_over_tau[index_kf, :])))
        plt.setp(line, ls ="-", c = 'k', lw = 3, marker = "", mfc = 'k', ms = 8, mec = "#7E2320", mew= 0)  # set properties

        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$t$", labelpad = 8)
        axes.set_ylabel(r"$\sum_{\rm t}$ $v_{\rm z}(t)$$e^{\rm \dfrac{-t}{\tau}}$", labelpad = 8)

        axes.locator_params(axis = 'y', nbins = 6)

        plt.show()

    #---------------------------------------------------------------------------

    def figParameters(self, fig_show=True):

        # (1,1) means one plot, and figsize is w x h in inch of figure
        fig, axes = plt.subplots(1, 1, figsize=(10.5, 5.8))
        # adjust the box of axes regarding the figure size
        fig.subplots_adjust(left=0.15, right=0.25, bottom=0.18, top=0.95)
        axes.remove()

        # Band name
        fig.text(0.45, 0.92, "Band #: " +
                    self.bandObject.bandname, fontsize=20, color='#2E00A4', style="italic")
        try:
            self.bandObject.M
            fig.text(0.41, 0.92, "AF", fontsize=20,
                        color="#FF0000", style="italic")
        except:
            None
        # Band Formulas
        fig.text(0.45, 0.47, "Band formula", fontsize=16,
                    color='#A9A9A9', style="italic")
        bandFormulaE2D = r"$\epsilon_{\rm k}^{\rm 2D}$ = - $\mu$" +\
            r" - 2$t$ (cos($k_{\rm x}a$) + cos($k_{\rm y}b$))" +\
            r" - 4$t^{'}$ (cos($k_{\rm x}a$) cos($k_{\rm y}b$))" + "\n" +\
            r"          - 2$t^{''}$ (cos(2$k_{\rm x}a$) + cos(2$k_{\rm y}b$))" + "\n"
        fig.text(0.45, 0.34, bandFormulaE2D, fontsize=12)

        bandFormulaEz = r"$\epsilon_{\rm k}^{\rm z}$   =" +\
            r" - 2$t_{\rm z}$ cos($k_{\rm z}c/2$) cos($k_{\rm x}a/2$) cos(2$k_{\rm y}b/2$) (cos($k_{\rm x}a$) - cos($k_{\rm y}b$))$^2$" + "\n" +\
            r"          - 2$t_{\rm z}^{'}$ cos($k_{\rm z}c/2$)"
        fig.text(0.45, 0.27, bandFormulaEz, fontsize=12)

        bandFormulaE3D = r"$\epsilon_{\rm k}^{\rm 3D}$   = $\epsilon_{\rm k}^{\rm 2D}$ + $\epsilon_{\rm k}^{\rm z}$"
        fig.text(0.45, 0.21, bandFormulaE3D, fontsize=12)

        # AF Band Formula
        try:
            self.bandObject.M
            if self.bandObject.electronPocket == True:
                sign_symbol = "+"
            else:
                sign_symbol = "-"
            AFBandFormula = r"$E_{\rm k}^{" + sign_symbol + r"}$ = 1/2 ($\epsilon_{\rm k}$ + $\epsilon_{\rm k+Q}$) " +\
                sign_symbol + \
                r" $\sqrt{1/4(\epsilon_{\rm k} - \epsilon_{\rm k+Q})^2 + \Delta_{\rm AF}^2}$ + $\epsilon_{\rm k}^{\rm z}$"
            fig.text(0.45, 0.15, AFBandFormula,
                        fontsize=12, color="#FF0000")
        except:
            None

        # Scattering Formula
        fig.text(0.45, 0.08, "Scattering formula",
                    fontsize=16, color='#A9A9A9', style="italic")
        scatteringFormula = r"$1 / \tau_{\rm tot}$ = $\Gamma_{\rm 0}$ + " + \
            r"$\Gamma_{\rm k}$ cos$^{\rm n}$(2$\phi$) + $\Gamma_{\rm DOS}^{\rm max}$ (|$v_{\rm k}^{\rm min}$| / |$v_{\rm k}$|)"
        fig.text(0.45, 0.03, scatteringFormula, fontsize=12)

        # Parameters Bandstructure
        fig.text(0.45, 0.85, "Band Parameters", fontsize=16,
                    color='#A9A9A9', style="italic")
        label_parameters = [r"$t$     =  " + "{0:.1f}".format(self.bandObject.t) + "    meV",
                            r"$\mu$    =  " +
                            "{0:+.3f}".format(self.bandObject.mu) +
                            r"   $t$",
                            r"$t^\prime$    =  " +
                            "{0:+.3f}".format(self.bandObject.tp) +
                            r"   $t$",
                            r"$t^{\prime\prime}$   =  " + "{0:+.3f}".format(
                                self.bandObject.tpp) + r"   $t$",
                            r"$t_{\rm z}$    =  " + "{0:+.3f}".format(
                                self.bandObject.tz) + r"   $t$",
                            r"$t_{\rm z}^{\prime}$    =  " +
                            "{0:+.3f}".format(self.bandObject.tz2) +
                            r"   $t$"
                            ]
        try:  # if it is a AF band
            self.bandObject.M
            label_parameters.append(
                r"$\Delta_{\rm AF}$ =  " + "{0:+.3f}".format(self.bandObject.M) + r"   $t$")
        except:
            None

        h_label = 0.80
        for label in label_parameters:
            fig.text(0.45, h_label, label, fontsize=14)
            h_label -= 0.04

        # Band filling
        fig.text(0.72, 0.85, "Band Filling =", fontsize=16,
                    color='#A9A9A9', style="italic")
        fig.text(0.855, 0.85, "{0:.3f}".format(
                    self.bandObject.n), fontsize=16, color='#73ea2b')

        # Scattering parameters
        fig.text(0.72, 0.77, "Scattering Parameters",
                    fontsize=16, color='#A9A9A9', style="italic")
        label_parameters = [
            r"$\Gamma_{\rm 0}$     = " + "{0:.1f}".format(self.gamma_0) +
            "   THz",
            r"$\Gamma_{\rm DOS}^{\rm max}$   = " +
            "{0:.1f}".format(self.gamma_dos_max) + "   THz",
            r"$\Gamma_{\rm k}$     = " + "{0:.1f}".format(self.gamma_k) +
            "   THz",
            r"$n$      = " + "{0:.0f}".format(self.power),
            r"$\Gamma_{\rm tot}^{\rm max}$   = " +
            "{0:.1f}".format(self.gamma_tot_max) + "   THz",
            r"$\Gamma_{\rm tot}^{\rm min}$   = " +
            "{0:.1f}".format(self.gamma_tot_min) + "   THz",
        ]
        h_label = 0.72
        for label in label_parameters:
            fig.text(0.72, h_label, label, fontsize=14)
            h_label -= 0.04

        ## Inset FS ///////////////////////////////////////////////////////////#
        a = self.bandObject.a
        b = self.bandObject.b
        c = self.bandObject.c

        mesh_graph = 201
        kx = np.linspace(-pi/a, pi/a, mesh_graph)
        ky = np.linspace(-pi/b, pi/b, mesh_graph)
        kxx, kyy = np.meshgrid(kx, ky, indexing='ij')

        axes_FS = plt.axes([-0.02, 0.56, .4, .4])
        axes_FS.set_aspect(aspect=1)
        axes_FS.contour(kxx, kyy, self.bandObject.e_3D_func(
            kxx, kyy, 0), 0, colors='#FF0000', linewidths=1)
        axes_FS.contour(kxx, kyy, self.bandObject.e_3D_func(
            kxx, kyy, pi / c), 0, colors='#00DC39', linewidths=1)
        axes_FS.contour(kxx, kyy, self.bandObject.e_3D_func(
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

        ## Inset Life Time ////////////////////////////////////////////////////#
        axes_tau = plt.axes([-0.02, 0.04, .4, .4])
        axes_tau.set_aspect(aspect=1)

        # phi = arctan2(ky, kx)
        # zeros = np.zeros_like(kx)
        # vx, vy, vz = self.bandObject.v_3D_func(kx, ky, 0)
        # print(np.min(vx))
        # print(np.min(vy))
        # print(np.min(vz))
        nn = int(self.bandObject.numberPointsPerKz_list[0] / 4)
        print(nn)
        kx = self.bandObject.kf[0, :nn]
        ky = self.bandObject.kf[1, :nn]
        phi = arctan2(ky, kx)
        vx = self.bandObject.vf[0, :nn]
        vy = self.bandObject.vf[1, :nn]
        vz = self.bandObject.vf[2, :nn]


        # nn = self.bandObject.numberPointsPerKz_list[0] # number of points at kz = 0
        # print(nn)
        gamma_kz0 = (self.gamma_0 +
                    self.gamma_DOS_Func(vx, vy, vz) +
                    self.gamma_k_Func(kx, ky) )
        # print(len(gamma_kz0))
        tau_kz0 = 1 / gamma_kz0
        tau_max_kz0 = np.max(tau_kz0)

        ## tau_0
        line = axes_tau.plot(cos(phi), sin(phi),
                                clip_on=False, label=r"$\tau_{\rm 0}$", zorder=10)
        plt.setp(line, ls="--", c='#000000', lw=1, marker="",
                    mfc='#000000', ms=5, mec="#7E2320", mew=0)
        ## tau_k
        line = axes_tau.plot(tau_kz0/tau_max_kz0*cos(phi), tau_kz0/tau_max_kz0*sin(phi),
                                clip_on=False, zorder=20, label=r"$\tau_{\rm tot}$")
        plt.setp(line, ls="-", c='#00FF9C', lw=1, marker="",
                    mfc='#000000', ms=5, mec="#7E2320", mew=0)
        # ## tau_k_min
        # phi_min = 3 * pi / 2
        # tau_k_x_min = 1 / \
        #     (gamma_0 + gamma_k * (sin(phi_min)**2 -
        #                             cos(phi_min)**2)**power) * cos(phi_min)
        # tau_k_y_min = 1 / \
        #     (gamma_0 + gamma_k * (sin(phi_min)**2 -
        #                             cos(phi_min)**2)**power) * sin(phi_min)
        # line = axes_tau.plot(tau_k_x_min / tau_0, tau_k_y_min / tau_0,
        #                         clip_on=False, label=r"$\tau_{\rm min}$", zorder=25)
        # plt.setp(line, ls="", c='#1CB7FF', lw=3, marker="o",
        #             mfc='#1CB7FF', ms=8, mec="#1CB7FF", mew=0)
        # ## tau_k_max
        # phi_max = 5 * pi / 4
        # tau_k_x_max = 1 / \
        #     (gamma_0 + gamma_k * (sin(phi_max)**2 -
        #                             cos(phi_max)**2)**power) * cos(phi_max)
        # tau_k_y_max = 1 / \
        #     (gamma_0 + gamma_k * (sin(phi_max)**2 -
        #                             cos(phi_max)**2)**power) * sin(phi_max)
        # line = axes_tau.plot(tau_k_x_max / tau_0, tau_k_y_max / tau_0,
        #                         clip_on=False, label=r"$\tau_{\rm max}$", zorder=25)
        # plt.setp(line, ls="", c='#FF8181', lw=3, marker="o",
        #             mfc='#FF8181', ms=8, mec="#FF8181", mew=0)

        # fraction = np.abs(np.round(tau_0 / tau_k_y_min, 2))
        # fig.text(0.30, 0.05, r"$\tau_{\rm max}$/$\tau_{\rm min}$" +
        #             "\n" + r"= {0:.1f}".format(fraction), fontsize=14)

        axes_tau.set_xlim(-1, 1)
        axes_tau.set_ylim(-1, 1)
        axes_tau.set_xticks([])
        axes_tau.set_yticks([])

        plt.legend(bbox_to_anchor=[1.49, 1.05], loc=1,
                    fontsize=14, frameon=False,
                    numpoints=1, markerscale=1, handletextpad=0.2)

        return fig

        ## Show figure ////////////////////////////////////////////////////////#
        if fig_show == True:
            plt.show()
        #//////////////////////////////////////////////////////////////////////////////#
