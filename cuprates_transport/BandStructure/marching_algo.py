import numpy as np
from numpy import pi, sqrt
from skimage import measure
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

## Marching algorithms ------------------------------------------------------#

def marching_cube(bandObj, epsilon=0):
    """
    Marching cube algorithm used to discretize the Fermi surface
    It returns: - kf with dimensions (xyz, position on FS)
                - dkf with dimensions (position on FS)
    """
    # Generate a uniform meshgrid
    e_3D, _, _, _, kx_a, ky_a, kz_a = bandObj.dispersion_grid(bandObj.res_xy,
        bandObj.res_xy, bandObj.res_z)
    # Use marching cubes to discritize the Fermi surface
    verts, faces, _, _ = measure.marching_cubes(e_3D,
            level=epsilon, spacing=(kx_a[1]-kx_a[0],
                                    ky_a[1]-ky_a[0],
                                    kz_a[1]-kz_a[0]),
                                    method='lewiner')
                                #---- Shape of verts is: (N_verts, 3)
                                #---- Shape of faces is: (N_faces, 3)
                                #---- Shape of triangles is: (N_faces, 3, 3)
                                #---- Shape of sides is: (N_faces, 2, 3)
                                #---- Shape of normal_vecs is: (N_faces, 3)
                                #---- Shape of areas is: (N_faces)
    # Recenter the Fermi surface after Marching Cube in the center of the BZ
    verts[:,0] = verts[:,0] - kx_a[-1]
    verts[:,1] = verts[:,1] - ky_a[-1]
    verts[:,2] = verts[:,2] - kz_a[-1]
    triangles = verts[faces]
    # Calculate areas
    sides = np.diff(triangles, axis=-2)
    # vectors that represent the sides of the triangles
    normal_vecs = np.cross(sides[...,0,:], sides[...,1,:])
    # cross product of two vectors
    # of the faces to calculate the areas of the triangles
    areas = np.linalg.norm(normal_vecs, axis=-1)/2
    # calculate the area of the triangles
    # by taking the norm of the cross product vector and divide by 2

    # Compute weight of each kf in surface integral
    dkf = np.zeros(len(verts))
    verts_repeated = faces.flatten() # shape is (3*N_faces)
    weights = np.repeat(areas/3, 3)  #1/3 for the volume of an irregular triangular prism
    dkf += np.bincount(verts_repeated, weights)
    kf = verts.transpose()
    return kf, dkf

def marching_square(bandObj, epsilon=0):
    """
    Marching square algorithm used to discretize the Fermi surface per slices
    It returns: - kf with dimensions (xyz, position on FS)
                - dkf with dimensions (position on FS)
    """
    # Initialize kx and ky arrays
    # res_xy_rough: make denser rough meshgrid to interpolate after
    if bandObj.a == bandObj.b:  # tetragonal case
        kx_a = np.linspace(0, pi / bandObj.a, bandObj.res_xy_rough)
        ky_a = np.linspace(0, pi / bandObj.b, bandObj.res_xy_rough)
    else:  # orthorhombic
        kx_a = np.linspace(-pi / bandObj.a, pi / bandObj.a, 2*bandObj.res_xy_rough)
        ky_a = np.linspace(-pi / bandObj.b, pi / bandObj.b, 2*bandObj.res_xy_rough)
    # Initialize kz array
    kz_a = np.linspace(-2 * pi / bandObj.c, 2 * pi / bandObj.c, bandObj.res_z)
    dkz = kz_a[1] - kz_a[0]  # integrand along z, in A^-1
    # Meshgrid for kx and ky
    kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')
    bandObj.number_of_points_per_kz_list = []
    # Loop over the kz array
    for j, kz in enumerate(kz_a):
        contours = measure.find_contours(bandObj.e_3D_func(kxx, kyy, kz), epsilon)
        number_of_points_per_kz = 0
        ## Loop over the different pieces of Fermi surfaces
        for i, contour in enumerate(contours):
            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            if bandObj.a == bandObj.b:
                x = contour[:, 0] / (bandObj.res_xy_rough - 1) * pi
                y = contour[:, 1] / (bandObj.res_xy_rough - 1) * pi / (bandObj.b / bandObj.a)  # anisotropy
            else:
                x = (contour[:, 0] / (2*bandObj.res_xy_rough - 1) - 0.5) * 2*pi
                y = (contour[:, 1] / (2*bandObj.res_xy_rough - 1) - 0.5) * 2*pi / (bandObj.b / bandObj.a) # anisotropy
            # path
            ds = sqrt(np.diff(x)**2 + np.diff(y)**2)  # segment lengths
            s = np.zeros_like(x)  # arrays of zeros
            s[1:] = np.cumsum(ds)  # integrate path, s[0] = 0

            number_of_points_on_contour = int(max(np.ceil(np.max(s) / (pi/bandObj.res_xy)), 4)) # choose at least a minimum of 4 points per contour
            number_of_points_per_kz += number_of_points_on_contour

            # interpolate and remove the last point (not to repeat)
            s_int = np.linspace(0, np.max(s), number_of_points_on_contour) # path
            dks = (s_int[1]- s_int[0]) / bandObj.a  # dk path
            x_int = np.interp(s_int, s, x)[:-1]
            y_int = np.interp(s_int, s, y)[:-1]
            if bandObj.a == bandObj.b:
                # For tetragonal symmetry, rotate the contour to get the entire Fermi surface
                x_dump = x_int
                y_dump = y_int
                for angle in [pi / 2, pi, 3 * pi / 2]:
                    x_int_p, y_int_p = bandObj.rotation(x_int, y_int, angle)
                    x_dump = np.append(x_dump, x_int_p)
                    y_dump = np.append(y_dump, y_int_p)
                x_int = x_dump
                y_int = y_dump
            # Put in an array /////////////////////////////////////////////////////#
            if i == 0 and j == 0:  # for first contour and first kz
                kxf = x_int / bandObj.a
                kyf = y_int / bandObj.b
                # bandObj.a (and not b) because anisotropy is taken into account earlier
                kzf = kz * np.ones_like(x_int)
                dkf = dks * dkz * np.ones_like(x_int)
                bandObj.dks = dks * np.ones_like(x_int)
                bandObj.dkz = dkz * np.ones_like(x_int)
            else:
                kxf = np.append(kxf, x_int / bandObj.a)
                kyf = np.append(kyf, y_int / bandObj.b)
                kzf = np.append(kzf, kz * np.ones_like(x_int))
                dkf = np.append(dkf, dks * dkz * np.ones_like(x_int))
                bandObj.dks = np.append(bandObj.dks, dks * np.ones_like(x_int))
                bandObj.dkz = np.append(bandObj.dkz, dkz * np.ones_like(x_int))
        if bandObj.a == bandObj.b:
            # discretize one fourth of FS, therefore need * 4
            bandObj.number_of_points_per_kz_list.append(4 * number_of_points_per_kz)
        else:
            bandObj.number_of_points_per_kz_list.append(number_of_points_per_kz)
    # dim -> (n, i0) = (xyz, position on FS)
    kf = np.vstack([kxf, kyf, kzf])
    return kf, dkf
