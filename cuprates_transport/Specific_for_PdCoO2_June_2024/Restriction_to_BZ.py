import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from skimage.measure import marching_cubes
from scipy.spatial import Delaunay
import time
import random as rd
import matplotlib as mpl

def generate_equidistant_points(point1, point2, density):

    # Convert points to numpy arrays for easier manipulation
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    # Generate linearly spaced numbers between 0 and 1
    t_values = np.linspace(0, 1, density + 2)  # +2 to include the endpoints
    
    # Interpolate points
    points = [p1 + t * (p2 - p1) for t in t_values]
    
    return points

def generate_point_cloud(BZ,n_points = 5):
    vertices_BZ = np.concatenate(BZ)
    cloud = []
    for i, v_i in enumerate(vertices_BZ):
        for j , v_j in enumerate(vertices_BZ):
            if not(i == j):
                cloud += [generate_equidistant_points(v_i,v_j,n_points)]
    return np.concatenate(cloud)

def print_BZ(ax, BZ,color = "k", alpha = 0.3):
    for face in BZ:
        vertices = np.array(face)
        # print(face)
        poly3d = Poly3DCollection([vertices])
        poly3d.setlpha(alpha)
        poly3d.set_edgecolor(color)
        poly3d.set_facecolor(color)
        # ax.scatter(face[0],face[1],face[2],c="r")
        ax.add_collection3d(poly3d)

def print_Delaunay(ax,BZ):
    bz = generate_point_cloud(BZ)
    delaunay = Delaunay(bz)
    tr = delaunay.simplices
    for simplex in tr:
        vertices = bz[simplex]
        # print(face)
        poly3d = Poly3DCollection([vertices])
        poly3d.set_alpha(0.3)
        poly3d.set_edgecolor('k')
        poly3d.set_facecolor('k')
        # ax.scatter(face[0],face[1],face[2],c="r")
        ax.add_collection3d(poly3d)

def get_Gvectors():

    # Define the ranges for n1, n2, and n3
    n1_range = [-4,-3,-2,-1,0,1,2,3,4]#np.concatenate((-np.arange(4),np.arange(4)))  # [0, 1, 2, 3]
    n2_range = [-4,-3,-2,-1,0,1,2,3,4]#np.concatenate((-np.arange(4),np.arange(4)))  # [0, 1, 2, 3]
    n3_range = [-4,-3,-2,-1,0,1,2,3,4]#np.concatenate((-np.arange(7),np.arange(7)))  # [0, 1, 2, 3, 4, 5, 6]

    # Create a meshgrid for the ranges
    n1, n2, n3 = np.meshgrid(n1_range, n2_range, n3_range, indexing='ij')

    # Stack the meshgrid arrays and reshape
    combined = np.stack((n1, n2, n3), axis=-1).reshape(-1, 3)

    modulus = np.linalg.norm(combined, axis=1)

    # Sort the array based on the modulus
    sorted_indices = np.argsort(modulus)
    sorted_combined = combined[sorted_indices]
    return sorted_combined



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# cif_file = "cif_files/CoPdO2_Prim_Cell.cif"
# structure = Structure.from_file(cif_file)

# sga = SpacegroupAnalyzer(structure)
# prim = sga.get_primitive_standard_structure()
# conv = sga.get_conventional_standard_structure()



def Fermi_Surface(prim,k_points,density_of_cloud = 5,Visualize_quiver = True,savefig =False):
    primBZ = prim.lattice.get_brillouin_zone()
    RL_basis = prim.lattice.reciprocal_lattice.matrix

    G_vectors = get_Gvectors()

    if Visualize_quiver:
        cmap = mpl.cm.get_cmap("magma", len(G_vectors))
        colors = cmap(np.arange(len(G_vectors)))
        np.random.shuffle(colors)

    FS = np.empty((0,3))

    cloud = []
    for i, bz in enumerate(G_vectors):
        bz_vertices = []
        for vertex in primBZ:
            bz_vertices.append(vertex + np.dot(RL_basis.T,bz))
    
        if k_points.size == 0:
            break
        print("\n Current Bz under investigation :", bz,"\n")
        
        cloud = generate_point_cloud(bz_vertices,n_points = density_of_cloud)
        print("cloud translated \n")

        new_bz = Delaunay(np.array(cloud))

        is_inside_newBZ = new_bz.find_simplex(k_points) >= 0
        print("New mask :  \n", is_inside_newBZ)

        new_FS = k_points[is_inside_newBZ]-np.dot(RL_basis.T,bz)

        if Visualize_quiver:
            if (new_FS.size !=0):
                print_BZ(ax,bz_vertices,color=colors[i],alpha=0.1)

            for p in k_points[is_inside_newBZ]:#new_FS:
                ax.scatter(*p,c=colors[i],alpha = 0.3)
                p_tr = p - np.dot(RL_basis.T,bz)
                dist = p_tr-p
                ax.scatter(*p_tr,c=colors[i])
                ax.quiver(*p,*dist,length = 1, color=colors[i], arrow_length_ratio=0.1)

        FS = np.concatenate((FS,new_FS))
        print("New FS : \n", FS)

        not_inside = np.array([not(el) for el in is_inside_newBZ])
        k_points = k_points[not_inside]

        print("Remaining points for investigation : \n", k_points)
    
    if Visualize_quiver:
        print_BZ(ax,primBZ,alpha = 0.2)
        # print_Delaunay(ax,primBZ)

        ax.view_init(elev=45, azim=30)
        ax.set_xlim([-2,2])
        ax.set_ylim([-2,2])
        ax.set_zlim([-2,2])
        if savefig:
            fig.savefig("Succesful_translation_to_1stBZ_5pts.pdf")
        plt.show()
    return FS

def generate_random_cloud():
    x = rd.uniform(-1.25,1.25)
    y = rd.uniform(-1.25,1.25)
    z = rd.uniform(-1.25,1.25)
    return np.array([x,y,z])

n_points = 25

k_points = np.array([generate_random_cloud() for p in range(n_points)])

def clipped_meshgrid_to_BZ(cif_file,res_x,res_y,res_z):
    
    ## Generation of the 1BZ
    structure = Structure.from_file(cif_file)

    sga = SpacegroupAnalyzer(structure)
    prim = sga.get_primitive_standard_structure()
    conv = sga.get_conventional_standard_structure()
    primBZ = prim.lattice.get_brillouin_zone()
    # primBZ = np.concatenate(primBZ)

    cloud = generate_point_cloud(primBZ,n_points = 5)
    
    new_bz = Delaunay(np.array(cloud))
    
    kx = np.linspace(-np.max(primBZ[:][0]),np.max(primBZ[:][0]),res_x)
    ky = np.linspace(-np.max(primBZ[:][1]),np.max(primBZ[:][1]),res_y)
    kz = np.linspace(-np.max(primBZ[:][2]),np.max(primBZ[:][2]),res_z)

    kxx, kyy, kzz = np.meshgrid(kx, ky, kz, indexing='ij')

    

    k_matrix_of_points = np.empty((res_x,res_y,res_z, 3))

    k_matrix_of_points[..., 0] = kxx
    k_matrix_of_points[..., 1] = kyy
    k_matrix_of_points[..., 2] = kzz

    print("Before_clipping : " ,k_matrix_of_points.shape, "\n",k_matrix_of_points)
    k_points = k_matrix_of_points.reshape(-1,3)
    insideBZ= new_bz.find_simplex(k_points) >= 0    
    k_points = k_points[insideBZ]
    print("After clipping : ", k_points.shape)
    kx = k_points[:][0]
    ky = k_points[:][1]
    kz = k_points[:][2]
    
    kxx, kyy, kzz = np.meshgrid(kx,ky,kz)
    e_3D = kxx**2 + kyy**2 + kzz**2
    return e_3D, kxx, kyy, kzz, kx, ky, kz


clipped_meshgrid_to_BZ("cuprates_transport\Specific_for_PdCoO2_June_2024\cif_files\CoPdO2_Prim_Cell.cif",10,10,10)