## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import pi, exp, sqrt, arctan2
from numba import jit, prange

from movement_equation import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Constant //////
hbar = 1.05e-34 # m2 kg / s
e = 1.6e-19 # C

## Units ////////
meVolt = 1.602e-22 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 10e-12 # 1 ps in seconds

units_chambers = e**2 / ( 4 * pi**3 ) * meVolt * picosecond / Angstrom / hbar**2
# this coefficient takes into accound all units and constant to prefactor Chambers formula

## Life time //////////////////////////////////////////////////////////////////#

@jit(nopython = True, cache = True)
def tOverTauFunc(k, v, tau_parameters, dt):
    gamma_0 = tau_parameters[0]
    gamma_k = tau_parameters[1]
    power   = tau_parameters[2]

    kx = k[0,:]
    ky = k[1,:]
    phi = arctan2(ky, kx)

    t_over_tau = np.cumsum(dt * ( gamma_0 + gamma_k * cos(2*phi)**power))

    return t_over_tau

## Chambers formula ///////////////////////////////////////////////////////////#
@jit(nopython = True, parallel = True)
def chambersFunc(kf, vf, dkf, kft, vft, t, tau_parameters):

    ## Velocity components
    vxf = vf[0,:]
    vyf = vf[1,:]
    vzf = vf[2,:]
    vzft = vft[2,:,:]

    # Time increment
    dt = t[1] - t[0]

    # Density of State
    dos = 1 / sqrt( vxf**2 + vyf**2 + vzf**2 )
            # = 1 / (hbar * |grad(E)|), here hbar is integrated in units_chambers

    # First the integral over time
    v_product = np.empty(vzf.shape[0], dtype = np.float64)
    for i0 in prange(vzf.shape[0]):
        vz_sum_over_t = np.sum( dos[i0] * vzft[i0,:] * exp( - tOverTauFunc(kft[:,i0,:], vft[:,i0,:], tau_parameters, dt) ) * dt ) # integral over t
        v_product[i0] = vzf[i0] * vz_sum_over_t # integral over z

    # Second the integral over kf
    sigma_zz = units_chambers * np.sum(dkf * v_product) # integral over k

    return sigma_zz


## rzz vs (B_theta, B_phi) ////////////////////////////////////////////////////#
def RzzAngleFunc(B_amp, B_theta_a, B_phi_a, kf, vf, dkf, band_parameters, tau_parameters):

    rho_zz_a = np.empty((B_phi_a.shape[0], B_theta_a.shape[0]), dtype = np.float64)
    gamma_0  = tau_parameters[0] # in THz
    tau_0    = 1 / gamma_0 # in picoseconds (1e-12 seconds)
    tmax     = 10 * tau_0

    for i in range(B_phi_a.shape[0]):
        for j in range(B_theta_a.shape[0]):
            kft, vft, t = solveMovementFunc(B_amp, B_theta_a[j], B_phi_a[i], kf, band_parameters, tmax)
            s_zz = chambersFunc(kf, vf, dkf, kft, vft, t, tau_parameters)
            rho_zz_a[i, j] = 1 / s_zz # dim (phi, theta)

    return rho_zz_a