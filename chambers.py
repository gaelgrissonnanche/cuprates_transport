import numpy as np
from numpy import cos, sin, pi, exp, sqrt, arctan2
from numba import jit, prange
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
        self.Bamp   = Bamp
        self.Btheta = Btheta
        self.Bphi   = Bphi
        self.B_vector = B_func(Bamp, Btheta, Bphi) # np array fo Bx,By,Bz

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

    def __eq__(self, other):
        return (
                self.Bamp       == other.Bamp
            and self.Btheta     == other.Btheta
            and self.Bphi       == other.Bphi
            and self.sigma   == other.sigma
        )

    def __ne__(self, other):
        return not self == other

    def solveMovementFunc(self):
        t_len = self.t.shape[0]
        kf_len = self.bandObject.kf.shape[1]
        kf = self.bandObject.kf.flatten() # flatten to get all the initial kf solved at the same time
        # Sovle differential equation
        kft = odeint(self.diffEqFunc, kf, self.t, rtol = 1e-4, atol = 1e-4).transpose()
        self.kft = np.reshape(kft, (3, kf_len, t_len))
        # Velocity function of time
        self.vft = np.empty_like(self.kft, dtype = np.float64)
        self.vft[0, :, :], self.vft[1, :, :], self.vft[2, :, :] = self.bandObject.v_3D_func(self.kft[0, :, :], self.kft[1, :, :], self.kft[2, :, :])

    def solveMovementForPoint(self, kpoint):
        t_len = self.t.shape[0]
        kt = odeint(self.diffEqFunc, kpoint, self.t, rtol = 1e-4, atol = 1e-4).transpose() # solve differential equation
        kt = np.reshape(kt, (3, 1, t_len))
        return kt

    def diffEqFunc(self, k, t):
        len_k = int(k.shape[0]/3)
        k.shape = (3, len_k) # reshape the flatten k
        vx, vy, vz =  self.bandObject.v_3D_func(k[0,:], k[1,:], k[2,:])
        dkdt = ( - units_move_eq ) * crossProductVectorized(vx, vy, vz, -self.B_vector[0], -self.B_vector[1], -self.B_vector[2])
                # (-) represent -t in vz(-t, k) in the Chambers formula
                # integrated from 0 to +infinity
        dkdt.shape = (3*len_k,) # flatten k again
        return dkdt

    def tOverTauFunc(self, k):
        phi = arctan2(k[1,:], k[0,:])
        # Integral from 0 to t of dt' / tau( k(t') ) or dt' / gamma( k(t') )
        t_over_tau = np.cumsum( self.dt * ( self.gamma_0 + self.gamma_k * cos(2*phi)**self.power) )
        return t_over_tau

    def VelocitiesProduct(self, i, j):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0: vif = vxf """
        ## Velocity components
        vif = self.bandObject.vf[i,:]
        vjft = self.vft[j,:,:]

        # Integral over time
        v_product = np.empty(vif.shape[0], dtype = np.float64)
        for i0 in prange(vif.shape[0]):
            vj_sum_over_t = np.sum( self.bandObject.dos[i0] * vjft[i0,:] * exp( - self.tOverTauFunc(self.kft[:,i0,:]) ) * self.dt ) # integral over t
            v_product[i0] = vif[i0] * vj_sum_over_t # integral over z

        return v_product

    def chambersFunc(self, i = 2, j = 2):
        """ Index i and j represent x, y, z = 0, 1, 2
            for example, if i = 0 and j = 1 : sigma[i,j] = sigma_xy """
        # The integral over kf
        self.sigma[i,j] = units_chambers * np.sum(self.bandObject.dkf * self.VelocitiesProduct(i = i, j = j))





# ABOUT JUST IN TIME (JIT) COMPILATION
# jitclass do not work, the best option is to call a jit otimized function from inside the class.
@jit(nopython=True, cache = True)
def B_func(B_amp, B_theta, B_phi):
    B = B_amp * np.array([sin(B_theta*pi/180)*cos(B_phi*pi/180), sin(B_theta*pi/180)*sin(B_phi*pi/180), cos(B_theta*pi/180)])
    return B

@jit("f8[:,:](f8[:], f8[:], f8[:], f8, f8, f8)", nopython=True, cache = True)
def crossProductVectorized(vx, vy, vz, Bx, By , Bz):
    product = np.empty((3, vx.shape[0]), dtype = np.float64)
    product[0,:] = vy[:] * Bz - vz[:] * By
    product[1,:] = vz[:] * Bx - vx[:] * Bz
    product[2,:] = vx[:] * By - vy[:] * Bx
    return product





