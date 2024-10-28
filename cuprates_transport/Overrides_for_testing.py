import numpy as np
from .conductivity_gammas import *
from cuprates_transport.conductivity import Conductivity
from cuprates_transport.bandstructure import BandStructure
from skimage.measure import marching_cubes
from cuprates_transport.translation_back_FBZ import translation_to_1stBZ

class BandStructure_FractionalSpace(BandStructure):
    

    def __init__(self,params):
        super().__init__(**params)

    def dispersion_grid(self, res_xy=500, res_z=100):        

        # Fractional space grid
        frac_kx = np.linspace(-0.5, 0.5, res_xy)
        frac_ky = np.linspace(-0.5, 0.5, res_xy)
        frac_kz = np.linspace(-0.5, 0.5, res_z)
        frac_kgrid = np.array(np.meshgrid(frac_kx, frac_ky, frac_kz))

        #!!!!!!!!!!! Important step here, which I don't understand. We must rotate the axis of the reciprocal basis by 90 degrees here if we want to achieve the correct surface. This is true for every dispersion. Maybe the Brillouin Zone is not well defined by pymatgen, or the reciprocal lattice vector are exchanged, or the transformation is not straightforward somehow. Still don't understand !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        def R(theta):
            theta = np.radians(theta)
            return [[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]

        cart_kgrid = np.matmul(np.matmul(self.prim_basis,R(90)).T,frac_kgrid.reshape(3,-1)).reshape(3,res_xy,res_xy,res_z) # Traslation to reciprocal space plus reshape to the form of a meshgrid again
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print('Overridden package being used')
        eps = self.e_3D_func(*cart_kgrid) # Evaluation in reciprocal space    
        return eps, frac_kgrid[0],frac_kgrid[1],frac_kgrid[2], frac_kx, frac_ky, frac_kz
    
    def marching_cube(self,epsilon):
        e_3, _, _, _, frac_kx, frac_ky, frac_kz = self.dispersion_grid(self.res_xy, self.res_z)
        

        verts, faces, _, _ = marching_cubes(e_3,
                level=epsilon, spacing=(frac_kx[1]-frac_kx[0],
                                        frac_ky[1]-frac_ky[0],
                                        frac_kz[1]-frac_kz[0]),
                                        method='lewiner')
        # Recenter the Fermi surface after Marching Cube in the center of the fractional space BZ
        verts[:,0] = verts[:,0] - frac_kx[-1]
        verts[:,1] = verts[:,1] - frac_ky[-1]
        verts[:,2] = verts[:,2] - frac_kz[-1]

        verts = np.matmul(verts,self.prim_basis)
        self.faces = faces
        

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
        
        # verts = translation_to_1stBZ(self.prim,verts)
        kf = verts.transpose()
        
        return kf, dkf

    def discretize_fermi_surface(self, epsilon=0):

        self.kf, self.dkf = self.marching_cube(epsilon)
        # Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0, :], self.kf[1, :], self.kf[2, :])
        # dim -> (n, i0) = (xyz, position on FS)
        self.vf = np.vstack([vx, vy, vz])
        # Density of State of k, dos_k in  meV^1 Angstrom^-1
        # dos_k = 1 / (|grad(E)|) here = 1 / (|v|),
        # because in the def of vf, hbar = 1
        self.dos_k = 1 / sqrt(self.vf[0,:]**2 + self.vf[1,:]**2 +self.vf[2,:]**2)

    def runBandStructure(self, epsilon=0, printDoping=False):
        self.discretize_fermi_surface(epsilon=epsilon)
        # self.doping(printDoping=printDoping)

