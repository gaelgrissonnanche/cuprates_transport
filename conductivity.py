import numpy as np
from numpy import cos, sin, pi, exp, sqrt, arctan2
from scipy.integrate import odeint
from skimage import measure
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
                 gamma_0=15, gamma_dos_max=0, gamma_k=0, power=2, az=0, factor_arcs=1):

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
        self.power   = power
        self.az      = az
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

    def factor_arcs_Func(self, kx, ky, kz):
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

    def gamma_DOS_Func(self, vx, vy, vz):
        dos = 1 / sqrt( vx**2 + vy**2 + vz**2 )
        dos_max = np.max(self.bandObject.dos)  # value to normalize the DOS to a quantity without units
        return self.gamma_dos_max * (dos / dos_max)

    def gamma_k_Func(self, kx, ky, kz):
        phi = arctan2(ky, kx)
        return self.gamma_k * np.abs(cos(2*phi))**self.power # / (1 + self.az*abs(sin(kz*self.bandObject.c/2)))

    def tOverTauFunc(self):
        # Integral from 0 to t of dt' / tau( k(t') ) or dt' * gamma( k(t') )
        self.t_over_tau = np.cumsum( self.dt_array / (
                                     self.tauTotFunc(self.kft[0, :, :],
                                                     self.kft[1, :, :],
                                                     self.kft[2, :, :],
                                                     self.vft[0, :, :],
                                                     self.vft[1, :, :],
                                                     self.vft[2, :, :]
                                                     )
                                                      )
                         , axis = 1)

    def tauTotFunc(self, kx, ky, kz, vx, vy, vz):
        """Computes the total lifetime based on the input model
        for the scattering rate"""

        #A = 0.5
        gammaTot = self.gamma_0 * np.ones_like(kx)
        if self.gamma_k!=0:
            gammaTot += self.gamma_k_Func(kx, ky, kz)   #*(1+A*np.abs(sin(kz*self.bandObject.c/2)))
        if self.gamma_dos_max!=0:
            gammaTot += self.gamma_DOS_Func(vx, vy, vz)
        if self.factor_arcs!=1:
            gammaTot *= self.factor_arcs_Func(kx, ky, kz)

        return 1/gammaTot

    def tauTotMaxFunc(self):
        # Compute the tau_max (the longest time between two collisions)
        # to better integrate from 0 --> 8 * 1 / gamma_min (instead of infinity)
        kf = self.bandObject.kf
        vf = self.bandObject.vf
        tauTotMax = np.max(self.tauTotFunc(kf[0, :], kf[1, :], kf[2, :],
                                           vf[0, :], vf[1, :], vf[2, :]))
        return tauTotMax

    def tauTotMinFunc(self):
        kf = self.bandObject.kf
        vf = self.bandObject.vf
        tauTotMin = np.min(self.tauTotFunc(kf[0, :], kf[1, :], kf[2, :],
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

    def figScatteringColor(self, kz=0, mesh_xy=501):
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 5.6))
        fig.subplots_adjust(left=0.10, right=0.85, bottom=0.20, top=0.9)

        kx_a = np.linspace(-pi / self.bandObject.a, pi / self.bandObject.a, mesh_xy)
        ky_a = np.linspace(-pi / self.bandObject.b, pi / self.bandObject.b, mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = self.bandObject.e_3D_func(kxx, kyy, kz)
        contours = measure.find_contours(bands, 0)

        gamma_max_list = []
        gamma_min_list = []
        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / self.bandObject.a
            ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / self.bandObject.b
            vx, vy, vz = self.bandObject.v_3D_func(kx, ky, kz)

            gamma_kz = 1 / self.tauTotFunc(kx, ky, kz, vx, vy, vz)
            gamma_max_list.append(np.max(gamma_kz))
            gamma_min_list.append(np.min(gamma_kz))

            points = np.array([kx*self.bandObject.a, ky*self.bandObject.b]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=plt.get_cmap('gnuplot'))
            lc.set_array(gamma_kz)
            lc.set_linewidth(4)

            line = axes.add_collection(lc)

        gamma_max = max(gamma_max_list)
        gamma_min = min(gamma_min_list)
        line.set_clim(gamma_min, gamma_max)
        cbar = fig.colorbar(line, ax=axes)
        cbar.minorticks_on()
        cbar.ax.set_ylabel(r'$\Gamma_{\rm tot}$ ( THz )', rotation=270, labelpad=40)

        kz = np.round(kz/(np.pi/self.bandObject.c), 1)
        fig.text(0.97, 0.05, r"$k_{\rm z}$ = " + "{0:g}".format(kz) + r"$\pi$/c", ha="right", color="r")
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
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 4.6))
        fig.subplots_adjust(left=0.20, right=0.8, bottom=0.20, top=0.9)
        axes2 = axes.twinx()
        axes2.set_axisbelow(True)

        ###
        kx_a = np.linspace(-pi / self.bandObject.a, pi / self.bandObject.a, mesh_xy)
        ky_a = np.linspace(-pi / self.bandObject.b, pi / self.bandObject.b, mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = self.bandObject.e_3D_func(kxx, kyy, kz)
        contours = measure.find_contours(bands, 0)

        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] / (mesh_xy - 1) -0.5) * 2*pi / self.bandObject.a
            ky = (contour[:, 1] / (mesh_xy - 1) -0.5) * 2*pi / self.bandObject.b
            vx, vy, vz = self.bandObject.v_3D_func(kx, ky, kz)

            gamma_kz = 1 / self.tauTotFunc(kx, ky, kz, vx, vy, vz)

            phi = np.rad2deg(np.arctan2(ky,kx))

            line = axes2.plot(phi, vz)
            plt.setp(line, ls ="", c = '#80ff80', lw = 3, marker = "o", mfc = '#80ff80', ms = 3, mec = "#7E2320", mew= 0)  # set properties

            line = axes.plot(phi, gamma_kz)
            plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 3, mec = "#7E2320", mew= 0)  # set properties

        axes.set_xlim(0, 180)
        axes.set_xticks([0, 45, 90, 135, 180])
        axes.tick_params(axis='x', which='major', pad=7)
        axes.tick_params(axis='y', which='major', pad=8)
        axes.set_xlabel(r"$\phi$", labelpad = 8)
        axes.set_ylabel(r"$\Gamma_{\rm tot}$ ( THz )", labelpad=8)
        axes2.set_ylabel(r"$v_{\rm z}$", rotation = 270, labelpad =25, color="#80ff80")



        kz = np.round(kz/(np.pi/self.bandObject.c), 1)
        fig.text(0.97, 0.05, r"$k_{\rm z}$ = " + "{0:g}".format(kz) + r"$\pi$/c", ha="right", color="r", fontsize = 20)
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
        scatteringFormula = r"$\Gamma_{\rm tot}$ = $\Gamma_{\rm 0}$ + " + \
            r"$\Gamma_{\rm k}$ |cos$^{\rm n}$(2$\phi$)| + $\Gamma_{\rm DOS}^{\rm max}$ (DOS / DOS$^{\rm max}$)"
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
            r"$n$      = " + "{0:g}".format(self.power),
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

        ## Inset Scattering rate ////////////////////////////////////////////////#
        axes_srate = plt.axes([-0.02, 0.04, .4, .4])
        axes_srate.set_aspect(aspect=1)

        mesh_xy = 501
        kx_a = np.linspace(-pi / self.bandObject.a, pi / self.bandObject.a,
                           mesh_xy)
        ky_a = np.linspace(-pi / self.bandObject.b, pi / self.bandObject.b,
                           mesh_xy)

        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')

        bands = self.bandObject.e_3D_func(kxx, kyy, 0)
        contours = measure.find_contours(bands, 0)

        gamma_max_list = []
        gamma_min_list = []
        for contour in contours:

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            kx = (contour[:, 0] /
                  (mesh_xy - 1) - 0.5) * 2 * pi / self.bandObject.a
            ky = (contour[:, 1] /
                  (mesh_xy - 1) - 0.5) * 2 * pi / self.bandObject.b
            vx, vy, vz = self.bandObject.v_3D_func(kx, ky, 0)

            gamma_kz0 = 1 / self.tauTotFunc(kx, ky, 0, vx, vy, vz)
            gamma_max_list.append(np.max(gamma_kz0))
            gamma_min_list.append(np.min(gamma_kz0))

            points = np.array([kx * self.bandObject.a,
                               ky * self.bandObject.b]).T.reshape(-1, 1, 2)
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

        return fig

        ## Show figure ////////////////////////////////////////////////////////#
        if fig_show == True:
            plt.show()
        #//////////////////////////////////////////////////////////////////////////////#
