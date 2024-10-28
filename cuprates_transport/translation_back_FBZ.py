import numpy as np
from scipy.constants import pi
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def generate_equidistant_points(point1, point2, density):

    """

    Interpolates linearly between two points and add a number of points in between equal to density
    
    """

    # Convert points to numpy arrays for easier manipulation
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    # Generate linearly spaced numbers between 0 and 1
    t_values = np.linspace(0, 1, density + 2)  # +2 to include the endpoints
    
    # Interpolate points
    points = [p1 + t * (p2 - p1) for t in t_values]
    
    return points

def generate_point_cloud(BZ,n_points = 5):

    """
    Generates a point of clouds to produce the Delaunay object. 
    BZ must be a SpacegroupAnalizer.get_primitive_standard_structure.lattice.get_brillouin_zone() object
    
    """

    vertices_BZ = np.concatenate(BZ)
    # vertices_BZ=BZ
    cloud = []
    for i, v_i in enumerate(vertices_BZ):
        for j , v_j in enumerate(vertices_BZ):
            if not(i == j):
                cloud += [generate_equidistant_points(v_i,v_j,n_points)]
    return np.concatenate(cloud)

def plot_cube_faces(ax):

    """
    This function plots a cube in axis ax. Used to plot the BZ in fractional space
    """

    # Define the vertices of the cube
    vertices = np.array([[-0.5, -0.5, -0.5],
                         [0.5, -0.5, -0.5],
                         [0.5, 0.5, -0.5],
                         [-0.5, 0.5, -0.5],
                         [-0.5, -0.5, 0.5],
                         [0.5, -0.5, 0.5],
                         [0.5, 0.5, 0.5],
                         [-0.5, 0.5, 0.5]])
    
    # Define the faces of the cube
    faces = [[vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]],
             [vertices[j] for j in [0, 1, 5, 4]],
             [vertices[j] for j in [2, 3, 7, 6]],
             [vertices[j] for j in [1, 2, 6, 5]],
             [vertices[j] for j in [4, 7, 3, 0]]]
    
    # Plot the faces
    poly3d = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
    ax.add_collection3d(poly3d)
    
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('kz')

def print_BZ(ax, BZ,color = "k", alpha = 0.3):
    """
    Prints the Brillouin Zone in the axis ax.
    BZ must be a SpacegroupAnalizer.get_primitive_standard_structure.lattice.get_brillouin_zone() object
    
    """
    BZ = BZ.lattice.get_brillouin_zone()
    for face in BZ:
        # face = rotate_points(face,90,axis ='z')
        vertices = np.array(face)
        # print(face)
        poly3d = Poly3DCollection([vertices])
        poly3d.set_alpha(alpha)
        poly3d.set_edgecolor(color)
        poly3d.set_facecolor(color)
        # ax.scatter(face[0],face[1],face[2],c="r")
        ax.add_collection3d(poly3d)

def get_Gvectors():

    """
    Translation vectors G(n1,n2,n3) = n1a* + n2b* + n3c*, where {a*,b*,c*} ar the reciprocal lattice vectors

    Output : 
    sorted_combined : An array of all posible combinations (n1,n2,n3)
    
    """

    # Define the ranges for n1, n2, and n3
    n1_range = [-4,-3,-2,-1,0,1,2,3,4]
    n2_range = [-4,-3,-2,-1,0,1,2,3,4]
    n3_range = [-4,-3,-2,-1,0,1,2,3,4]

    # Create a meshgrid for the ranges
    n1, n2, n3 = np.meshgrid(n1_range, n2_range, n3_range, indexing='ij')

    # Stack the meshgrid arrays and reshape
    combined = np.stack((n1, n2, n3), axis=-1).reshape(-1, 3)

    modulus = np.linalg.norm(combined, axis=1)

    # Sort the array based on the modulus
    sorted_indices = np.argsort(modulus)
    sorted_combined = combined[sorted_indices]
    return sorted_combined

def translation_to_1stBZ(prim,k_points,density_of_cloud = 5):
    """
    This function translates an arbitrary set of points k_points to the 1st Brillouin Zone (BZ) defined by the prim parameter
    
    Inputs :
    prim -> This is a SpacegroupAnalizer.get_primitive_standard_structure object. We obtain the BZ boundaries and the reciprocal lattice matrix from here.
    k_points -> Set of points to translate
    density_of_cloud -> Parameter that set the precision with which the BZ is investigated (goes into the cloud of points generating the Delaunay object). 5 is enough generally.

    Output:
    FS -> Translated set of points
    
    """



    primBZ = prim.lattice.get_brillouin_zone()
    RL_basis = prim.lattice.reciprocal_lattice.matrix
    
    G_vectors = get_Gvectors()

    initial_n_points = len(k_points)

    FS = np.empty((0,3))

    cloud = []
    for i, bz in enumerate(G_vectors):
        """
        For all combinations of n1, n2, n3, i.e. for all BZ's, calculates which points are inside, translates them back to the first using G(n1,n2,n3), and removes them from the set of points k_points. Continues until there are no points left. Stores the translated points into an array FS

        """


        bz_vertices = []

        for vertex in primBZ:
            bz_vertices.append(vertex + np.dot(RL_basis.T,bz)) ## All vertices of the new displaced BZ
        
        #~~~~~~~~~Trying to make it faster~~~~~~~~~~~~~~~~~~~~
        # flattened_list = [array for sublist in primBZ for array in sublist]
        # bz_vertices = flattened_list + np.dot(RL_basis.T,bz) #!!! ATTENTION, if you do this, remove the concatenate inside the generate_point_cloud() function !!!!!!!!!!

        ## I think it makes it slower...
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if k_points.size == 0:
            break
        # print("\n Current Bz under investigation :", bz,"\n")
        
        cloud = generate_point_cloud(bz_vertices,n_points = density_of_cloud)
        # print("cloud translated \n")
        # print(cloud)
        new_bz = Delaunay(np.array(cloud)) ## Generates the space where points must lie

        is_inside_newBZ = new_bz.find_simplex(k_points) >= 0
        # print("New mask :  \n", is_inside_newBZ)

        new_FS = k_points[is_inside_newBZ]-np.dot(RL_basis.T,bz)
        
        FS = np.concatenate((FS,new_FS))
        # print("New FS : \n", FS)

        not_inside = np.array([not(el) for el in is_inside_newBZ])
        k_points = k_points[not_inside]

        # print("Remaining points for investigation : \n", len(k_points), "/",initial_n_points)

    return FS
