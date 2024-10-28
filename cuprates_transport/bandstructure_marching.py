import numpy as np
from numpy import pi, sqrt
from scipy.constants import electron_mass, physical_constants
from skimage import measure
from scipy.spatial import Delaunay
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

# Constant //////
hbar = 1  # velocity will be in units of 1 / hbar,
# this hbar is taken into accound in the constant units_move_eq

# Units ////////
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule
m0 = electron_mass # in kg
Angstrom = 1e-10  # 1 A in meters

# cif_file = "cif_files/CoPdO2_Prim_Cell.cif"

def marching_cube(self, epsilon=0):
    """
    Marching cube algorithm used to discretize the Fermi surface
    It returns: - kf with dimensions (xyz, position on FS)
                - dkf with dimensions (position on FS)
    """
    # Generate a uniform meshgrid
    e_3D, _, _, _, kx_a, ky_a, kz_a = self.dispersion_grid(self.res_xy,
        self.res_xy, self.res_z)
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

def marching_square(self, epsilon=0):
    """
    Marching square algorithm used to discretize the Fermi surface per slices
    It returns: - kf with dimensions (xyz, position on FS)
                - dkf with dimensions (position on FS)
    """
    # Initialize kx and ky arrays
    # res_xy_rough: make denser rough meshgrid to interpolate after
    if self.a == self.b:  # tetragonal case
        kx_a = np.linspace(0, pi / self.a, self.res_xy_rough)
        ky_a = np.linspace(0, pi / self.b, self.res_xy_rough)
    else:  # orthorhombic
        kx_a = np.linspace(-pi / self.a, pi / self.a, 2*self.res_xy_rough)
        ky_a = np.linspace(-pi / self.b, pi / self.b, 2*self.res_xy_rough)
    # Initialize kz array
    kz_a = np.linspace(-2 * pi / self.c, 2 * pi / self.c, self.res_z)
    dkz = kz_a[1] - kz_a[0]  # integrand along z, in A^-1
    # Meshgrid for kx and ky
    kxx, kyy = np.meshgrid(kx_a, ky_a, indexing='ij')
    # Loop over the kz array
    for j, kz in enumerate(kz_a):
        contours = measure.find_contours(self.e_3D_func(kxx, kyy, kz), epsilon)
        number_of_points_per_kz = 0
        ## Loop over the different pieces of Fermi surfaces
        for i, contour in enumerate(contours):
            # Contour come in units proportionnal to size of meshgrid
            # one want to scale to units of kx and ky
            if self.a == self.b:
                x = contour[:, 0] / (self.res_xy_rough - 1) * pi
                y = contour[:, 1] / (self.res_xy_rough - 1) * pi / (self.b / self.a)  # anisotropy
            else:
                x = (contour[:, 0] / (2*self.res_xy_rough - 1) - 0.5) * 2*pi
                y = (contour[:, 1] / (2*self.res_xy_rough - 1) - 0.5) * 2*pi / (self.b / self.a) # anisotropy
            # path
            ds = sqrt(np.diff(x)**2 + np.diff(y)**2)  # segment lengths
            s = np.zeros_like(x)  # arrays of zeros
            s[1:] = np.cumsum(ds)  # integrate path, s[0] = 0

            number_of_points_on_contour = int(max(np.ceil(np.max(s) / (pi/self.res_xy)), 4)) # choose at least a minimum of 4 points per contour
            number_of_points_per_kz += number_of_points_on_contour

            # interpolate and remove the last point (not to repeat)
            s_int = np.linspace(0, np.max(s), number_of_points_on_contour) # path
            dks = (s_int[1]- s_int[0]) / self.a  # dk path
            x_int = np.interp(s_int, s, x)[:-1]
            y_int = np.interp(s_int, s, y)[:-1]
            if self.a == self.b:
                # For tetragonal symmetry, rotate the contour to get the entire Fermi surface
                x_dump = x_int
                y_dump = y_int
                for angle in [pi / 2, pi, 3 * pi / 2]:
                    x_int_p, y_int_p = self.rotation(x_int, y_int, angle)
                    x_dump = np.append(x_dump, x_int_p)
                    y_dump = np.append(y_dump, y_int_p)
                x_int = x_dump
                y_int = y_dump
            # Put in an array /////////////////////////////////////////////////////#
            if i == 0 and j == 0:  # for first contour and first kz
                kxf = x_int / self.a
                kyf = y_int / self.b
                # self.a (and not b) because anisotropy is taken into account earlier
                kzf = kz * np.ones_like(x_int)
                dkf = dks * dkz * np.ones_like(x_int)
                self.dks = dks * np.ones_like(x_int)
                self.dkz = dkz * np.ones_like(x_int)
            else:
                kxf = np.append(kxf, x_int / self.a)
                kyf = np.append(kyf, y_int / self.b)
                kzf = np.append(kzf, kz * np.ones_like(x_int))
                dkf = np.append(dkf, dks * dkz * np.ones_like(x_int))
                self.dks = np.append(self.dks, dks * np.ones_like(x_int))
                self.dkz = np.append(self.dkz, dkz * np.ones_like(x_int))
        if self.a == self.b:
            # discretize one fourth of FS, therefore need * 4
            self.number_of_points_per_kz_list.append(4 * number_of_points_per_kz)
        else:
            self.number_of_points_per_kz_list.append(number_of_points_per_kz)
    # dim -> (n, i0) = (xyz, position on FS)
    kf = np.vstack([kxf, kyf, kzf])
    return kf, dkf
