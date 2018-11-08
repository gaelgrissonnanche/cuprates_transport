import numpy as np
from numpy import cos, sin, pi, exp, sqrt, arctan2
from numba import jit, prange
from scipy.integrate import odeint

from band import BandStructure

hbar = 1.05e-34 # m2 kg / s
e = 1.6e-19 # C
meVolt = 1.602e-22 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 10e-12 # 1 ps in seconds
units_move_eq =  e * Angstrom**2 * picosecond * meVolt / hbar**2 # this coefficient takes into accound all units and constant to prefactor the movement equation
units_chambers = e**2 / ( 4 * pi**3 ) * meVolt * picosecond / Angstrom / hbar**2    # this coefficient takes into accound all units and constant to prefactor Chambers formula


@jit(nopython=True, cache = True)
def B_func(B_amp, B_theta, B_phi):
    B = B_amp * np.array([sin(B_theta)*cos(B_phi), sin(B_theta)*sin(B_phi), cos(B_theta)])
    return B

@jit("f8[:,:](f8[:], f8[:], f8[:], f8, f8, f8)", nopython=True, cache = True)
def cross_product_vectorized(vx, vy, vz, Bx, By , Bz):
    product = np.empty((3, vx.shape[0]), dtype = np.float64)
    product[0,:] = vy[:] * Bz - vz[:] * By
    product[1,:] = vz[:] * Bx - vx[:] * Bz
    product[2,:] = vx[:] * By - vy[:] * Bx
    return product


class amroPoint:
    def __init__(self, band, Bamp, Btheta, Bphi):

        self.bandObject = band ## WARNING do not modify within this object
        self.Bamp = Bamp
        self.Btheta = Btheta
        self.Bphi = Bphi
        self.B = B_func(Bamp, Btheta, Bphi) # np array fo Bx,By,Bz
        self.gamma_0 = 152 # in THz
        self.gamma_k = 649 # in THz
        self.power   = 12

        self.power = int(self.power)
        if self.power % 2 == 1:
            self.power += 1
            tau_parameters[2] = self.power

        self.tau_0 = 1 / self.gamma_0
        self.tmax = 10*self.tau_0
        self.Ntime = 300
        self.dt = self.tmax / self.Ntime
        self.t = np.arange(0, self.tmax, self.dt)

        self.kft = None
        self.vft = None
        self.sigma_zz = None

    def __eq__(self, other):
        return (
                self.Bamp       == other.Bamp 
            and self.Btheta     == other.Btheta
            and self.Bphi       == other.Bphi
            and self.sigma_zz   == other.sigma_zz
        )
    
    def __ne__(self, other):
        return not self == other

    def solveMovementFunc(self):
        
        t_len = self.t.shape[0]
        kf_len = self.bandObject.kf.shape[1]
        kf = self.bandObject.kf.flatten() # flatten to get all the initial kf solved at the same time
        kft = odeint(self.diff_func_vectorized, kf, self.t, rtol = 1e-4, atol = 1e-4).transpose() # solve differential equation
        
        self.kft = np.reshape(kft, (3, kf_len, t_len))
        self.vft = np.empty_like(self.kft, dtype = np.float64)
        self.vft[0, :, :], self.vft[1, :, :], self.vft[2, :, :] = self.bandObject.v_3D_func(self.kft[0, :, :], self.kft[1, :, :], self.kft[2, :, :])

    def solveMovementForPoint(self, kpoint):
        t_len = self.t.shape[0]
        kt = odeint(self.diff_func_vectorized, kpoint, self.t, rtol = 1e-4, atol = 1e-4).transpose() # solve differential equation
        kt = np.reshape(kft, (3, 1, t_len))
        return kt

    #@jit(nopython = True, cache = True)
    def diff_func_vectorized(self, k, t):
        len_k = int(k.shape[0]/3)
        k = np.reshape(k, (3, len_k))  ##### WARNING HERE THERE IS A COPY
        vx, vy, vz =  self.bandObject.v_3D_func(k[0,:], k[1,:], k[2,:])
        dkdt = ( - units_move_eq ) * cross_product_vectorized(vx, vy, vz, -self.B[0], -self.B[1], -self.B[2]) # (-) represent -t in vz(-t, k) in the Chambers formula
                                # integrated from 0 to +infinity
        dkdt = dkdt.flatten()
        return dkdt

    #@jit(nopython = True, cache = True)
    def tOverTauFunc(self, k):
        kx = k[0,:]
        ky = k[1,:]
        phi = arctan2(ky, kx)
        t_over_tau = np.cumsum(self.dt * ( self.gamma_0 + self.gamma_k * cos(2*phi)**self.power))
        return t_over_tau

    # def computeAMRO(self):
    #@jit(nopython = True, parallel = True)
    def vz_product(self):

        ## Velocity components
        vxf = self.bandObject.vf[0,:]
        vyf = self.bandObject.vf[1,:]
        vzf = self.bandObject.vf[2,:]
        vzft = self.vft[2,:,:]

        # Time increment
        dt = self.t[1] - self.t[0]
        # Density of State
        dos = sqrt( vxf**2 + vyf**2 + vzf**2 )
        # = 1 / (hbar * |grad(E)|), here hbar is integrated in units_chambers  ### WARNING is ther '1/' missing?

        # First the integral over time
        v_product = np.empty(vzf.shape[0], dtype = np.float64)
        for i0 in prange(vzf.shape[0]):
            vz_sum_over_t = np.sum( ( 1 / dos[i0] ) * vzft[i0,:] * exp( - self.tOverTauFunc(self.kft[:,i0,:]) ) * self.dt ) # integral over t
            v_product[i0] = vzf[i0] * vz_sum_over_t # integral over z

        return v_product
        # Second the integral over kf

    def chambersFunc(self):
        self.sigma_zz = units_chambers * np.sum(self.bandObject.dkf * self.vz_product()) # integral over k













