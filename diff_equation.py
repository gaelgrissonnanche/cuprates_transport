# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin
from numba import jit, prange
from scipy.integrate import odeint
from diffeqpy import de

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


## Solve differential equation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# @jit(nopython = True, cache = True)
def solve_movement_func(B_amp, B_theta, B_phi, kf, band_parameters, tmax):

    dt = tmax / 300
    t = np.arange(0, tmax, dt)

    ## Compute B ////#
    B = B_func(B_amp, B_theta, B_phi)

    ## Run solver ///#
    # kft = rgk4_algorithm(kf, t, B, band_parameters)
    # vft = np.empty_like(kft, dtype = np.float64)
    # vft[0,:,:], vft[1,:,:], vft[2,:,:] = v_3D_func(kft[0,:,:], kft[1,:,:], kft[2,:,:], band_parameters)

    # kf_len = kf.shape[1]
    # t_len = t.shape[0]
    # # Compute kf, vf function of t ///#
    # kf = kf.flatten()
    # kft = odeint(diff_func_vectorized_odeint, kf, t, args = (B, band_parameters)).transpose() # solve differential equation
    # kft = np.reshape(kft, (3, kf_len, t_len))
    # vft = np.empty_like(kft, dtype = np.float64)
    # vft[0, :, :], vft[1, :, :], vft[2, :, :] = v_3D_func(kft[0, :, :], kft[1, :, :], kft[2, :, :], band_parameters)

    kf_len = kf.shape[1]

    # Compute kf, vf function of t ///#
    tspan = (0, tmax)
    p = np.append(B, band_parameters)
    kf = kf.flatten()

    prob = de.ODEProblem(diff_func_vectorized_julia, kf, tspan, p)
    sol = de.solve(prob, saveat = dt, abstol = 1e-3, reltol=1e-3)
    t = np.array(sol.t)
    t_len = t.shape[0]
    kft = np.array(sol.u).transpose()
    kft = np.reshape(kft, (3, kf_len, t_len))
    vft = np.empty_like(kft, dtype = np.float64)
    vft[0, :, :], vft[1, :, :], vft[2, :, :] = v_3D_func(kft[0, :, :], kft[1, :, :], kft[2, :, :], band_parameters)

    return kft, vft, t


# Old stuff for ODEINT ///////////////////////////////////////////////////////#

@jit(nopython=True, cache = True)
def diff_func_vectorized_odeint(k, t, B, band_parameters):
    len_k = int(k.shape[0]/3)
    k = np.reshape(k, (3, len_k))
    vx, vy, vz =  v_3D_func(k[0,:], k[1,:], k[2,:], band_parameters)
    dkdt = ( - e / hbar ) * cross_product_vectorized(vx, vy, vz, -B[0], -B[1], -B[2]) # (-) represent -t in vz(-t, k) in the Chambers formula
                            # integrated from 0 to +infinity
    dkdt = dkdt.flatten()
    return dkdt

@jit(nopython=True, cache = True)
def diff_func_vectorized_julia(k, p, t):
    B = p[:3]
    band_parameters = p[3:]
    len_k = int(k.shape[0]/3)
    k = np.reshape(k, (3, len_k))
    vx, vy, vz =  v_3D_func(k[0,:], k[1,:], k[2,:], band_parameters)
    dkdt = ( - e / hbar ) * cross_product_vectorized(vx, vy, vz, -B[0], -B[1], -B[2]) # (-) represent -t in vz(-t, k) in the Chambers formula
                            # integrated from 0 to +infinity
    dkdt = dkdt.flatten()
    return dkdt

# kft = np.empty( (3, kf.shape[1], t.shape[0]), dtype = np.float64) # dim -> (n, i0, i) = (xyz, position on FS @ t= 0, position on FS after ->t)
# vft = np.empty_like(kft, dtype = np.float64)

# # Compute kf, vf function of t ///#
# for i0 in range(kf.shape[1]):
#     # prob = de.ODEProblem(f, u0, tspan)
#     # de.solve(prob)
#     kft[:, i0, :] = odeint(diff_func, kf[:, i0], t, args = (B, band_parameters)).transpose() # solve differential equation
#     vft[0, i0, :], vft[1, i0, :], vft[2, i0, :] = v_3D_func(kft[0, i0, :], kft[1, i0, :], kft[2, i0, :], band_parameters)
#     print(i0)
