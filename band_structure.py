# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin
from numba import jit
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Constant //////
# hbar = 1.05e34
# e = 1.6e19
# m0 = 9.1e31
# kB = 1.38e23
# jtoev = 6.242e18
e = 1
hbar = 1
m = 1


## Band structure /////////////////////////////////////////////////////////////#
@jit(nopython=True)
def e_2D_func(kx, ky, mu, a, t, tp, tpp):
    e_2D = -mu + 2 * t * ( cos(kx*a) + cos(ky*a) ) + 4 * tp * cos(kx*a) * cos(ky*a) + 2 * tpp * ( cos(2*kx*a) + cos(2*ky*a) )
    return e_2D

@jit(nopython=True)
def e_z_func(kx, ky, kz, tz, a, d):
    sigma = cos(kx*a/2) * cos(ky*a/2)
    e_z = 2 * tz * sigma * ( cos(kx*a) - cos(ky*a) )**2 * cos(kz*d)
    return e_z

@jit(nopython=True)
def e_3D_func(kx, ky, kz, band_parameters):
    mu = band_parameters[0]
    a = band_parameters[1]
    d = band_parameters[2]
    t = band_parameters[3]
    tp = band_parameters[4]
    tpp = band_parameters[5]
    tz = band_parameters[6]

    e_3D = e_2D_func(kx, ky, mu, a, t, tp, tpp) + \
           e_z_func(kx, ky, kz, tz, a, d)
    return e_3D

@jit(nopython=True)
def e_3D_func_radial(r, theta, kz, band_parameters):
    kx = r * cos(theta)
    ky = r * sin(theta)
    return e_3D_func(kx, ky, kz, band_parameters)

@jit(nopython=True)
def v_3D_func(kx, ky, kz, band_parameters):
    a = band_parameters[1]
    d = band_parameters[2]
    t = band_parameters[3]
    tp = band_parameters[4]
    tpp = band_parameters[5]
    tz = band_parameters[6]

    # Velocity from e_2D
    d_e2D_dkx = -2 * t * a * sin(kx*a) - 4 * tp * a * sin(kx*a)*cos(ky*a) - 4 * tpp * a * sin(2*kx*a)
    d_e2D_dky = -2 * t * a * sin(ky*a) - 4 * tp * a * cos(kx*a)*sin(ky*a) - 4 * tpp * a * sin(2*ky*a)
    d_e2D_dkz = 0

    # Velocity from e_z
    sigma = cos(kx*a/2) * cos(ky*a/2)
    d_sigma_dkx = - a / 2 * sin(kx*a/2) * cos(ky*a/2)
    d_sigma_dky = - a / 2 * cos(kx*a/2) * sin(ky*a/2)

    d_ez_dkx = 2 * tz * d_sigma_dkx * (cos(kx*a) - cos(ky*a))**2 * cos(kz*d) + \
               2 * tz * sigma * 2 * (cos(kx*a) - cos(ky*a)) * (-a * sin(kx*a)) * cos(kz*d)
    d_ez_dky = 2 * tz * d_sigma_dky * (cos(kx*a) - cos(ky*a))**2 * cos(kz*d) + \
               2 * tz * sigma * 2 * (cos(kx*a) - cos(ky*a)) * (+a * sin(ky*a)) * cos(kz*d)
    d_ez_dkz = 2 * tz * sigma * (cos(kx*a) - cos(ky*a))**2 * (-d * sin(kz*d))

    vx = d_e2D_dkx + d_ez_dkx
    vy = d_e2D_dky + d_ez_dky
    vz = d_e2D_dkz + d_ez_dkz

    return vx, vy, vz

