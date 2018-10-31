## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin
from numba import jit
from scipy.integrate import odeint

from band_structure import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Constant //////
hbar = 1.05e-34 # m2 kg / s
e = 1.6e-19 # C

## Units ////////
meVolt = 1.602e-22 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 10e-12 # 1 ps in seconds

units_move_eq =  e * Angstrom**2 * picosecond * meVolt / hbar**2
# this coefficient takes into accound all units and constant to prefactor the movement equation

## Magnetic field function ////////////////////////////////////////////////////#
@jit(nopython=True, cache = True)
def B_func(B_amp, B_theta, B_phi):
    B = B_amp * np.array([sin(B_theta)*cos(B_phi), sin(B_theta)*sin(B_phi), cos(B_theta)])
    return B

## Cross product //////////////////////////////////////////////////////////////#
@jit("f8[:,:](f8[:], f8[:], f8[:], f8, f8, f8)", nopython=True, cache = True)
def cross_product_vectorized(vx, vy, vz, Bx, By , Bz):
    product = np.empty((3, vx.shape[0]), dtype = np.float64)
    product[0,:] = vy[:] * Bz - vz[:] * By
    product[1,:] = vz[:] * Bx - vx[:] * Bz
    product[2,:] = vx[:] * By - vy[:] * Bx
    return product

## Differential equation function /////////////////////////////////////////////#
@jit(nopython = True, cache = True)
def diff_func_vectorized(k, t, B, band_parameters):
    len_k = int(k.shape[0]/3)
    k = np.reshape(k, (3, len_k))
    vx, vy, vz =  v_3D_func(k[0,:], k[1,:], k[2,:], band_parameters)
    dkdt = ( - units_move_eq ) * cross_product_vectorized(vx, vy, vz, -B[0], -B[1], -B[2]) # (-) represent -t in vz(-t, k) in the Chambers formula
                            # integrated from 0 to +infinity
    dkdt = dkdt.flatten()
    return dkdt

## Solve differential equation ////////////////////////////////////////////////#
def solveMovementFunc(B_amp, B_theta, B_phi, kf, band_parameters, tmax):

    dt = tmax / 300
    t = np.arange(0, tmax, dt)

    ## Compute B ////#
    B = B_func(B_amp, B_theta, B_phi)

    ## Run solver ///#
    kf_len = kf.shape[1]
    t_len = t.shape[0]
    kf = kf.flatten() # flatten to get all the initial kf solved at the same time
    kft = odeint(diff_func_vectorized, kf, t, args = (B, band_parameters), rtol = 1e-4, atol = 1e-4).transpose() # solve differential equation
    kft = np.reshape(kft, (3, kf_len, t_len))
    vft = np.empty_like(kft, dtype = np.float64)
    vft[0, :, :], vft[1, :, :], vft[2, :, :] = v_3D_func(kft[0, :, :], kft[1, :, :], kft[2, :, :], band_parameters)

    return kft, vft, t



