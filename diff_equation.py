## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin
from numba import jit, prange

from band_structure import *
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#


## Constant //////
# hbar = 1.05e34
# e = 1.6e19
# m0 = 9.1e31
# kB = 1.38e23

e = 1
hbar = 1
m = 1


## Magnetic field function ////////////////////////////////////////////////////#
@jit(nopython=True, cache = True)
def B_func(B_amp, B_theta, B_phi):
    B = B_amp * np.array([sin(B_theta)*cos(B_phi), sin(B_theta)*sin(B_phi), cos(B_theta)])
    return B

## Functions for Runge-Kutta //////////////////////////////////////////////////#
@jit("f8[:,:](f8[:], f8[:], f8[:], f8, f8, f8)", nopython=True, cache = True)
def cross_product_vectorized(vx, vy, vz, Bx, By , Bz):
    product = np.empty((3, vx.shape[0]), dtype = np.float64)
    product[0,:] = vy[:] * Bz - vz[:] * By
    product[1,:] = vz[:] * Bx - vx[:] * Bz
    product[2,:] = vx[:] * By - vy[:] * Bx
    return product

@jit("f8[:,:](f8[:,:], f8, f8[:], f8[:])", nopython=True, cache = True)
def diff_func_vectorized(k, t, B, band_parameters):
    vx, vy, vz =  v_3D_func(k[0,:], k[1,:], k[2,:], band_parameters)
    dkdt = ( - e / hbar ) * cross_product_vectorized(vx, vy, vz, -B[0], -B[1], -B[2]) # (-) represent -t in vz(-t, k) in the Chambers formula
                            # integrated from 0 to +infinity
    return dkdt

@jit("f8[:,:,:](f8[:,:], f8[:], f8[:], f8[:])", nopython=True, parallel = True)
def rgk4_algorithm(kf, t, B, band_parameters):

    dt = t[1] - t[0]
    kft = np.empty( (3, kf.shape[1], t.shape[0]), dtype = np.float64) # dim -> (n, i0, i) = (xyz, position on FS @ t= 0, position on FS after ->t)

    k = kf.astype(np.float64) # initial value of k for Runge-Kutta
    ## this copy to float64 type helps the k += operation in the memory
    for i in range(t.shape[0]):
        k1 = dt * diff_func_vectorized(k, t[i], B, band_parameters)
        k2 = dt * diff_func_vectorized(k + k1/2, t[i] + dt/2, B, band_parameters)
        k3 = dt * diff_func_vectorized(k + k2/2, t[i] + dt/2, B, band_parameters)
        k4 = dt * diff_func_vectorized(k + k3, t[i] + dt, B, band_parameters)
        k += (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4
        kft[:, :, i] = k

    return kft



## Old stuff for ODEINT ///////////////////////////////////////////////////////#

# @jit("f8[:](f8[:], f8[:])", nopython=True, cache = True)
# def cross_product(u, v):
#     product = empty(u.shape[0])
#     product[0] = u[1] * v[2] - u[2] * v[1]
#     product[1] = u[2] * v[0] - u[0] * v[2]
#     product[2] = u[0] * v[1] - u[1] * v[0]
#     return product

# ## Movement equation //#
# @jit("f8[:](f8[:], f8, f8[:], f8[:])", nopython=True, cache = True)
# def diff_func(k, t, B, band_parameters):
#     vx, vy, vz =  v_3D_func(k[0], k[1], k[2], band_parameters)
#     v = array([vx, vy, vz]).transpose()
#     dkdt = ( - e / hbar ) * cross_product(v, - B) # (-) represent -t in vz(-t, k) in the Chambers formula
#                             # integrated from 0 to +infinity
#     return dkdt

# ## Compute kf, vf function of t ///#
# for i0 in range(kf.shape[0]):
#     kft[i0, :, :] = odeint(diff_func, kf[i0, :], t, args = (B, band_parameters)) # solve differential equation
#     vx, vy, vz = v_3D_func(kft[i0, :, 0], kft[i0, :, 1], kft[i0, :, 2], band_parameters)
#     vft[i0, :, :] = np.array([vx, vy, vz]).transpose()