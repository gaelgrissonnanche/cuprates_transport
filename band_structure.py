## Modules <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
import numpy as np
from numpy import cos, sin, pi
from skimage import measure
from numba import jit
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Constant //////
hbar = 1 # velocity will be in units of 1 / hbar,
         # this hbar is taken into accound in the constant units_move_eq

## Band structure /////////////////////////////////////////////////////////////#
@jit(nopython = True, cache = True)
def e_2D_func(kx, ky, a, b, mu, t, tp, tpp):
    e_2D = -mu + 2 * t * ( cos(kx*a) + cos(ky*b) ) + 4 * tp * cos(kx*a) * cos(ky*b) + 2 * tpp * ( cos(2*kx*a) + cos(2*ky*b) )
    return e_2D

@jit(nopython = True, cache = True)
def e_z_func(kx, ky, kz, a, b, c, tz, tz2):
    d = c / 2.
    sigma = cos(kx*a/2) * cos(ky*b/2)
    e_z = 2 * tz * sigma * ( cos(kx*a) - cos(ky*b) )**2 * cos(kz*d) + 2 * tz2 * cos(kz*d)
    return e_z

@jit(nopython = True, cache = True)
def e_3D_func(kx, ky, kz, band_parameters):
    a = band_parameters[0]
    b = band_parameters[1]
    c = band_parameters[2]
    mu = band_parameters[3]
    t = band_parameters[4]
    tp = band_parameters[5]
    tpp = band_parameters[6]
    tz = band_parameters[7]
    tz2 = band_parameters[8]

    e_3D = e_2D_func(kx, ky, a, b, mu, t, tp, tpp) + \
           e_z_func(kx, ky, kz, a, b, c, tz, tz2)
    return e_3D


@jit(nopython = True, cache = True)
def v_3D_func(kx, ky, kz, band_parameters):
    a = band_parameters[0]
    b = band_parameters[1]
    c = band_parameters[2]
    d = c / 2
    t = band_parameters[4]
    tp = band_parameters[5]
    tpp = band_parameters[6]
    tz = band_parameters[7]
    tz2 = band_parameters[8]

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
    d_ez_dkz = 2 * tz * sigma * (cos(kx*a) - cos(ky*b))**2 * (-d * sin(kz*d)) + 2 * tz2 * (-d) * sin(kz*d)

    vx = ( 1 / hbar ) * (d_e2D_dkx + d_ez_dkx)
    vy = ( 1 / hbar ) * (d_e2D_dky + d_ez_dky)
    vz = ( 1 / hbar ) * (d_e2D_dkz + d_ez_dkz)

    return vx, vy, vz

## Discretizing FS function ///////////////////////////////////////////////////#
def rotation(x, y, angle):
    xp =  cos(angle)*x + sin(angle)*y
    yp = -sin(angle)*x + cos(angle)*y
    return xp, yp

def discretize_FS(band_parameters, mesh_parameters):

    a = band_parameters[0]
    b = band_parameters[1]
    c = band_parameters[2]

    mesh_xy   = mesh_parameters[0]
    mesh_z    = mesh_parameters[1]

    mesh_xy_rough = mesh_xy * 10 + 1 # make denser rough meshgrid to interpolate

    kz_a = np.linspace(0, 2*pi/c, mesh_z) # half of FBZ, 2*pi/c because bodycentered unit cell
    kx_a = np.linspace(0, pi/a, mesh_xy_rough)
    ky_a = np.linspace(0, pi/b, mesh_xy_rough)
    kxx, kyy = np.meshgrid(kx_a, ky_a, indexing = 'ij')

    for j, kz in enumerate(kz_a):
        bands = e_3D_func(kxx, kyy, kz, band_parameters)
        contours = measure.find_contours(bands, 0)
        number_contours = len(contours)

        for i, contour in enumerate(contours):

            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            x = contour[:, 0]/(mesh_xy_rough-1)*pi/a
            y = contour[:, 1]/(mesh_xy_rough-1)*pi/b

            ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2) # segment lengths
            s = np.zeros_like(x) # arrays of zeros
            s[1:] = np.cumsum(ds) # integrate path, s[0] = 0
            dkf_weight = s.max() / (mesh_xy + 1) # weight to ponderate dkf

            s_int = np.linspace(0, s.max(), mesh_xy + 1) # regular spaced path, add one
            x_int = np.interp(s_int, s, x)[:-1] # interpolate and remove the last point (not to repeat)
            y_int = np.interp(s_int, s, y)[:-1]

            ## Rotate the contour to get the entire Fermi surface
            x_dump = x_int
            y_dump = y_int
            for angle in [pi/2, pi, 3*pi/2]:
                x_int_p, y_int_p = rotation(x_int, y_int, angle)
                x_dump = np.append(x_dump, x_int_p)
                y_dump = np.append(y_dump, y_int_p)
            x_int = x_dump
            y_int = y_dump

            # Put in an array /////////////////////////////////////////////////////#
            if i == 0 and j == 0: # for first contour and first kz
                kxf = x_int
                kyf = y_int
                kzf = kz * np.ones_like(x_int)
                dkf = dkf_weight * np.ones_like(x_int)
            else:
                kxf = np.append(kxf, x_int)
                kyf = np.append(kyf, y_int)
                kzf = np.append(kzf, kz*np.ones_like(x_int))
                dkf = np.append(dkf, dkf_weight * np.ones_like(x_int))

    kf = np.vstack([kxf, kyf, kzf]) # dim -> (n, i0) = (xyz, position on FS)

    ## Integration Delta
    # dkf = 2 / (mesh_xy * mesh_z) * ( 2 * pi )**2 / ( a * b ) * ( 4 * pi ) / c

    ## Compute Velocity at t = 0 on Fermi Surface
    vx, vy, vz = v_3D_func(kf[0,:], kf[1,:], kf[2,:], band_parameters)
    vf = np.vstack([vx, vy, vz]) # dim -> (i, i0) = (xyz, position on FS)

    return kf, vf, dkf, number_contours

## Hole doping function ///////////////////////////////////////////////////////#
def dopingFunc(band_parameters):
    a = band_parameters[0]
    b = band_parameters[1]
    c = band_parameters[2]

    kx_a = np.linspace(-pi/a, pi/a, 500)
    ky_a = np.linspace(-pi/b, pi/b, 500)
    kz_a = np.linspace(-2*pi/c, 2*pi/c, 10)
    kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a, indexing = 'ij')
    E = - e_3D_func(kxx, kyy, kzz, band_parameters) # (-) because band is inverted in my conventions

    # Number of k in the total Brillouin Zone
    N = E.shape[0] * E.shape[1] * E.shape[2]
    # Number of k in the Brillouin zone per plane
    N_per_plane = E.shape[0] * E.shape[1]
    # Number of electron in the total Brillouin Zone
    n = 2 / N * np.sum( np.greater_equal(0, E) ) # number of quasiparticles below mu, 2 is for the spin
    p = 1 - n # number of holes
    # Number of electron in the Brillouin zone per plane
    n_per_plane = 2 / N_per_plane * np.sum( np.greater_equal(0, E), axis = (0,1) )
    p_per_plane = 1 - n_per_plane

    return p, p_per_plane

