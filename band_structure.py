# -*- coding: Latin-1 -*-

## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin, pi
from skimage import measure
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
def e_2D_func(kx, ky, a, b, mu, t, tp, tpp):
    e_2D = -mu + 2 * t * ( cos(kx*a) + cos(ky*b) ) + 4 * tp * cos(kx*a) * cos(ky*b) + 2 * tpp * ( cos(2*kx*a) + cos(2*ky*b) )
    return e_2D

@jit(nopython=True)
def e_z_func(kx, ky, kz, a, b, c, tz):
    d = c / 2.
    sigma = cos(kx*a/2) * cos(ky*b/2)
    e_z = 2 * tz * sigma * ( cos(kx*a) - cos(ky*b) )**2 * cos(kz*d)
    return e_z

@jit(nopython=True)
def e_3D_func(kx, ky, kz, band_parameters):
    a = band_parameters[0]
    b = band_parameters[1]
    c = band_parameters[2]
    mu = band_parameters[3]
    t = band_parameters[4]
    tp = band_parameters[5]
    tpp = band_parameters[6]
    tz = band_parameters[7]

    e_3D = e_2D_func(kx, ky, a, b, mu, t, tp, tpp) + \
           e_z_func(kx, ky, kz, a, b, c, tz)
    return e_3D


@jit(nopython=True)
def v_3D_func(kx, ky, kz, band_parameters):
    a = band_parameters[0]
    b = band_parameters[1]
    c = band_parameters[2]
    d = c / 2
    t = band_parameters[4]
    tp = band_parameters[5]
    tpp = band_parameters[6]
    tz = band_parameters[7]

    # Velocity from e_2D
    d_e2D_dkx = -2 * t * a * sin(kx*a) - 4 * tp * a * sin(kx*a)*cos(ky*b) - 4 * tpp * a * sin(2*kx*a)
    d_e2D_dky = -2 * t * b * sin(ky*b) - 4 * tp * b * cos(kx*a)*sin(ky*b) - 4 * tpp * b * sin(2*ky*b)
    d_e2D_dkz = 0

    # Velocity from e_z
    sigma = cos(kx*a/2) * cos(ky*b/2)
    d_sigma_dkx = - a / 2 * sin(kx*a/2) * cos(ky*b/2)
    d_sigma_dky = - b / 2 * cos(kx*a/2) * sin(ky*b/2)

    d_ez_dkx = 2 * tz * d_sigma_dkx * (cos(kx*a) - cos(ky*b))**2 * cos(kz*d) + \
               2 * tz * sigma * 2 * (cos(kx*a) - cos(ky*b)) * (-a * sin(kx*a)) * cos(kz*d)
    d_ez_dky = 2 * tz * d_sigma_dky * (cos(kx*a) - cos(ky*b))**2 * cos(kz*d) + \
               2 * tz * sigma * 2 * (cos(kx*a) - cos(ky*b)) * (+b * sin(ky*b)) * cos(kz*d)
    d_ez_dkz = 2 * tz * sigma * (cos(kx*a) - cos(ky*b))**2 * (-d * sin(kz*d))

    vx = d_e2D_dkx + d_ez_dkx
    vy = d_e2D_dky + d_ez_dky
    vz = d_e2D_dkz + d_ez_dkz

    return vx, vy, vz

## Discretizing FS function
def discretize_FS(band_parameters, mesh_xy, mesh_z, half_FS_z):

    a = band_parameters[0]
    b = band_parameters[1]
    c = band_parameters[2]

    mesh_xy_rough = mesh_xy * 10 + 1 # make denser rough meshgrid to interpolate

    if half_FS_z == True:
        kz_a = np.linspace(0, 2*pi/c, mesh_z) # 2*pi/c because bodycentered unit cell
    else:
        kz_a = np.linspace(-2*pi/c, 2*pi/c, mesh_z) # 2*pi/c because bodycentered unit cell

    kx_a = np.linspace(-pi/a, pi/a, mesh_xy_rough)
    ky_a = np.linspace(-pi/b, pi/b, mesh_xy_rough)
    kxx, kyy = np.meshgrid(kx_a, ky_a, indexing = 'ij')

    for j, kz in enumerate(kz_a):
        bands = e_3D_func(kxx, kyy, kz, band_parameters)
        contours = measure.find_contours(bands, 0)
        number_contours = len(contours)

        for i, contour in enumerate(contours):

            # Contour in units proportionnal to size of meshgrid
            x_raw = contour[:, 0]
            y_raw = contour[:, 1]

            # Is Contour closed?
            closed = (x_raw[0] == x_raw[-1]) * (y_raw[0] == y_raw[-1])

            # Scale the contour to units of kx and ky
            x = (x_raw/(mesh_xy_rough-1)-0.5)*2*pi/a
            y = (y_raw/(mesh_xy_rough-1)-0.5)*2*pi/b

            # Make all opened contours go in direction of x[i+1] - x[i] < 0, otherwise interpolation problem
            if closed == False and np.diff(x)[0] > 0:
                x = x[::-1]
                y = y[::-1]

            # Make the contour start at a high point of symmetry, for example for ky = 0
            index_xmax = np.argmax(x) # find the index of the first maximum of x
            x = np.roll(x, x.shape - index_xmax) # roll the elements to get maximum of x first
            y = np.roll(y, x.shape - index_xmax) # roll the elements to get maximum of x first

            # Closed contour
            if closed == True: # meaning a closed contour
                x = np.append(x, x[0]) # add the first element to get a closed contour
                y = np.append(y, y[0]) # in order to calculate its real total length
                mesh_xy = mesh_xy - (mesh_xy % 4) # respects the 4-order symmetry

                ds = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
                s = np.zeros_like(x) # arrays of zeros
                s[1:] = np.cumsum(ds) # integrate path, s[0] = 0

                s_int = np.linspace(0, s.max(), mesh_xy + 1) # regular spaced path
                x_int = np.interp(s_int, s, x)[:-1] # interpolate
                y_int = np.interp(s_int, s, y)[:-1]

            # Opened contour
            else:
                ds = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
                s = np.zeros_like(x) # arrays of zeros
                s[1:] = np.cumsum(ds) # integrate path, s[0] = 0

                s_int = np.linspace(0, s.max(), mesh_xy) # regular spaced path
                x_int = np.interp(s_int, s, x) # interpolate
                y_int = np.interp(s_int, s, y)


            # Put in an array /////////////////////////////////////////////////////#
            if i == 0 and j == 0: # for first contour and first kz
                kxf = x_int
                kyf = y_int
                kzf = kz*np.ones_like(x_int)
            else:
                kxf = np.append(kxf, x_int)
                kyf = np.append(kyf, y_int)
                kzf = np.append(kzf, kz*np.ones_like(x_int))

    kf = np.vstack([kxf, kyf, kzf]) # dim -> (n, i0) = (xyz, position on FS)

    ## Integration Delta
    if half_FS_z == True:
        dkf = 1 / (mesh_xy * mesh_z) * ( 2 * pi )**3 / ( a * b * c ) * 2
    else:
        dkf = 1 / (mesh_xy * mesh_z) * ( 2 * pi )**3 / ( a * b * c )

    ## Compute Velocity at t = 0 on Fermi Surface
    vx, vy, vz = v_3D_func(kf[0,:], kf[1,:], kf[2,:], band_parameters)
    vf = np.vstack([vx, vy, vz]) # dim -> (i, i0) = (xyz, position on FS)

    return kf, vf, dkf, number_contours

