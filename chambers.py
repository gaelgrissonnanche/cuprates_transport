import numpy as np
from numpy import cos, sin, pi, exp, sqrt, arctan2
from scipy.integrate import odeint
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Constant //////
hbar = 1.05e-34 # m2 kg / s
e = 1.6e-19 # C

## Units ////////
meVolt = 1.602e-22 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 10e-12 # 1 ps in seconds

## This coefficient takes into accound all units and constant to prefactor the movement equation
units_move_eq =  e * Angstrom**2 * picosecond * meVolt / hbar**2
## This coefficient takes into accound all units and constant to prefactor Chambers formula
units_chambers = e**2 / ( 4 * pi**3 ) * meVolt * picosecond / Angstrom / hbar**2


class Conductivity:
    def __init__(self, bandObject, Bamp, Bphi, Btheta, gamma_0=152, gamma_k=649, power=12):

        # Band object
        self.bandObject = bandObject ## WARNING do not modify within this object

        # Magnetic field
        self._Bamp   = Bamp
        self._Btheta = Btheta
        self._Bphi   = Bphi
        self._B_vector = self.BFunc() # np array fo Bx,By,Bz

        # Scattering rate
        self.gamma_0 = gamma_0 # in THz
        self.gamma_k = gamma_k # in THz
        self.power   = int(power)
        if self.power % 2 == 1:
            self.power += 1

        # Time parameters
        self.tau_0 = 1 / self.gamma_0 # in picoseconds
        self.tmax = 10 * self.tau_0 # in picoseconds
        self.Ntime = 300 # number of steps in time
        self.dt = self.tmax / self.Ntime
        self.t = np.arange(0, self.tmax, self.dt)

        # Time-dependent kf, vf
        self.kft = None
        self.vft = None

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

    ## Special Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def __eq__(self, other):
        return (
                self._Bamp       == other._Bamp
            and self._Btheta     == other._Btheta
            and self._Bphi       == other._Bphi
            and self.sigma   == other.sigma
        )

    def __ne__(self, other):
        return not self == other

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def BFunc(self):
        B = self._Bamp * \
            np.array([sin(self._Btheta*pi/180)*cos(self._Bphi*pi/180),
                      sin(self._Btheta*pi/180)*sin(self._Bphi*pi/180),
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

    def solveMovementForPoint(self, kpoint):
        len_t = self.t.shape[0]
        kt = odeint(self.diffEqFunc, kpoint, self.t, rtol = 1e-4, atol = 1e-4).transpose() # solve differential equation
        kt = np.reshape(kt, (3, 1, len_t))
        return kt

    def diffEqFunc(self, k, t):
        len_k = int(k.shape[0]/3)
        k.shape = (3, len_k) # reshape the flatten k
        vx, vy, vz =  self.bandObject.v_3D_func(k[0,:], k[1,:], k[2,:])
        dkdt = ( - units_move_eq ) * self.crossProductVectorized(vx, vy, vz)

        dkdt.shape = (3*len_k,) # flatten k again
        return dkdt

    def tOverTauFunc(self, k):
        phi = arctan2(k[1,:], k[0,:])
        # Integral from 0 to t of dt' / tau( k(t') ) or dt' / gamma( k(t') )
        t_over_tau = np.cumsum( self.dt * ( self.gamma_0 + self.gamma_k * cos(2*phi)**self.power) )
        return t_over_tau

    # def tOverTauFunc(self, k):
    #     phi = arctan2(k[1,:], k[0,:])
    #     # Integral from 0 to t of dt' / tau( k(t') ) or dt' / gamma( k(t') )
    #     t_over_tau = np.cumsum( self.dt / ( 0.026 - 0.0135 * cos(4*phi)) )
    #     return t_over_tau

    def VelocitiesProduct(self, i, j):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0: vif = vxf """
        ## Velocity components
        vif  = self.bandObject.vf[i,:]
        vjft = self.vft[j,:,:]

        # Integral over time
        v_product = np.empty(vif.shape[0], dtype = np.float64)
        for i0 in range(vif.shape[0]):
            vj_sum_over_t = np.sum( self.bandObject.dos[i0] * vjft[i0,:] * exp( - self.tOverTauFunc(self.kft[:,i0,:]) ) * self.dt ) # integral over t
            v_product[i0] = vif[i0] * vj_sum_over_t # integral over z

        return v_product

    def chambersFunc(self, i = 2, j = 2):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0 and j = 1 : sigma[i,j] = sigma_xy """
        # The integral over kf
        self.sigma[i,j] = units_chambers * np.sum(self.bandObject.dkf * self.VelocitiesProduct(i = i, j = j))





