# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin
from numba import jit
import time

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
# @jit(nopython=True, cache = True)
def B_func(B_amp, B_theta, B_phi):
    B = B_amp * np.array([sin(B_theta)*cos(B_phi), sin(B_theta)*sin(B_phi), cos(B_theta)])
    return B

## Functions for Runge-Kutta //////////////////////////////////////////////////#
@jit("f8[:,:,:](f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:])", nopython=True, cache = True)
def cross_product_vectorized(vx, vy, vz, Bx, By , Bz):
    dkdt = np.empty((3, vx.shape[0], Bx.shape[1]), dtype = np.float64)

    dkdt[0,:,:] = vy[:,:] * Bz[:,:] - vz[:,:] * By[:,:]
    dkdt[1,:,:] = vz[:,:] * Bx[:,:] - vx[:,:] * Bz[:,:]
    dkdt[2,:,:] = vx[:,:] * By[:,:] - vy[:,:] * Bx[:,:]

    return dkdt

@jit("f8[:,:,:](f8[:,:,:], f8, f8[:,:,:], f8[:])", nopython=True, cache = True)
def diff_func_vectorized(k, t, B_flat_2D_repeated, band_parameters):
    # start_diff = time.time()
    vx, vy, vz =  v_3D_func(k[0,:,:], k[1,:,:], k[2,:,:], band_parameters)
    # print("time diff function: %.6s seconds" % (time.time() - start_diff))
    dkdt = ( - e / hbar ) * cross_product_vectorized(vx, vy, vz, -B_flat_2D_repeated[0,:,:], -B_flat_2D_repeated[1,:,:], -B_flat_2D_repeated[2,:,:]) # (-) represent -t in vz(-t, k) in the Chambers formula
                            # integrated from 0 to +infinity
    return dkdt

@jit("f8[:,:,:,:](f8[:,:], f8[:], f8[:,:,:], f8[:])", nopython=True, cache = True, nogil = True)
def rgk4_algorithm(kf, t, B_tensor, band_parameters):

    dt = t[1] - t[0]
    kft = np.empty( (3, kf.shape[1], t.shape[0], B_tensor.shape[1] * B_tensor.shape[2]), dtype = np.float64) # dim -> (n, i0, i) = (xyz, position on FS @ t= 0, position on FS after ->t)

    Bx_flat_1D = B_tensor[0,:,:].flatten() # put all rows one after the other in a one-dimension
    By_flat_1D = B_tensor[1,:,:].flatten() # array of size size_theta * size_phi, to go back to
    Bz_flat_1D = B_tensor[2,:,:].flatten() # the original, use B[n,:,:] = Bn.reshape(B_theta_aa.shape)

    kfx = np.outer(kf[0, :], np.ones_like(Bx_flat_1D)) # kfx : matrix of size (len(kfx) x len(Bx))
    kfy = np.outer(kf[1, :], np.ones_like(By_flat_1D)) # if (i , j) are the index, each row
    kfz = np.outer(kf[2, :], np.ones_like(Bz_flat_1D)) # has the same value of kf[i] repeated at each column corresping to index of Bx[j]


    k = np.dstack((kfx, kfy, kfz)).transpose((2,0,1))

    ## Make three 2D arrays for Bx, By, Bz where for each arrays, each rows is
    ## the same Bx_flat_1D array (or equivalent By_, Bz_)
    Bx_2D_repeated = np.outer(np.ones(kf.shape[1]), Bx_flat_1D) #
    By_2D_repeated = np.outer(np.ones(kf.shape[1]), By_flat_1D)
    Bz_2D_repeated = np.outer(np.ones(kf.shape[1]), Bz_flat_1D)

    B_2D_repeated = np.dstack((Bx_2D_repeated, By_2D_repeated, Bz_2D_repeated)).transpose((2,0,1))


    for i in range(t.shape[0]):
        k1 = dt * diff_func_vectorized(k, t[i], B_2D_repeated, band_parameters)
        k2 = dt * diff_func_vectorized(k + k1/2, t[i] + dt/2, B_2D_repeated, band_parameters)
        k3 = dt * diff_func_vectorized(k + k2/2, t[i] + dt/2, B_2D_repeated, band_parameters)
        k4 = dt * diff_func_vectorized(k + k3, t[i] + dt, B_2D_repeated, band_parameters)
        k += (1/6)*k1 + (1/3)*k2 + (1/3)*k3 + (1/6)*k4
        kft[:, :, i, :] = k

        print("i = ", i)


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