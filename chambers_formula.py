## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import pi, exp, sqrt, arctan2
from numba import jit, prange

from movement_equation import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


## Constant //////
# hbar = 1.05e-34 # m2 kg / s
# e = 1.6e-19 # C
# m0 = 9.1e-31 # kg

e = 1
hbar = 1
m0 = 1

## Life time //////////////////////////////////////////////////////////////////#
@jit(nopython = True, cache = True)
def tauFunc(k, tau_parameters):
    tau_0   = tau_parameters[0]
    gamma_0 = 1 / tau_0
    gamma_k = tau_parameters[1]
    power   = tau_parameters[2]

    kx = k[0,:]
    ky = k[1,:]
    phi = arctan2(ky, kx)

    tau = 1 / ( gamma_0 + gamma_k * cos(2*phi)**power)

    return tau


## Chambers formula ///////////////////////////////////////////////////////////#
@jit(nopython = True, parallel = True)
def chambersFunc(kf, vf, dkf, kft, vft, t, tau_parameters):

    prefactor = e**2 / ( 4 * pi**3 )

    ## Velocity components
    vxf = vf[0,:]
    vyf = vf[1,:]
    vzf = vf[2,:]
    vzft = vft[2,:,:]

    # Time increment
    dt = t[1] - t[0]
    # Density of State
    dos = hbar * sqrt( vxf**2 + vyf**2 + vzf**2 )

    # First the integral over time
    v_product = np.empty(vzf.shape[0], dtype = np.float64)
    for i0 in prange(vzf.shape[0]):
        vz_sum_over_t = np.sum( ( 1 / dos[i0] ) * vzft[i0,:] * exp( - t / tauFunc(kft[:,i0,:], tau_parameters) ) * dt ) # integral over t
        v_product[i0] = vzf[i0] * vz_sum_over_t # integral over z

    # Second the integral over kf
    sigma_zz = prefactor * np.sum(dkf * v_product) # integral over k

    return sigma_zz


## rzz vs (B_theta, B_phi) ////////////////////////////////////////////////////#
def admrFunc(B_amp, B_theta_a, B_phi_a, kf, vf, dkf, band_parameters, tau_parameters):

    rho_zz_a = np.empty((B_phi_a.shape[0], B_theta_a.shape[0]), dtype = np.float64)
    tau_0    = tau_parameters[0]
    tmax     = 10 * tau_0

    for i in range(B_phi_a.shape[0]):
        for j in range(B_theta_a.shape[0]):
            kft, vft, t = solveMovementFunc(B_amp, B_theta_a[j], B_phi_a[i], kf, band_parameters, tmax)
            s_zz = chambersFunc(kf, vf, dkf, kft, vft, t, tau_parameters)
            rho_zz_a[i, j] = 1 / s_zz # dim (phi, theta)

    return rho_zz_a