import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from mpl_toolkits.mplot3d import Axes3D
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #

# ///// RC Parameters ////// #
mpl.rcdefaults()
mpl.rcParams['font.size'] = 24.         # Fontsize
mpl.rcParams['font.family'] = 'Arial'   # Font Arial
mpl.rcParams['axes.labelsize'] = 24.
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.width'] = 0.6
mpl.rcParams['ytick.major.width'] = 0.6
mpl.rcParams['axes.linewidth'] = 0.6    # Thickness of the axes lines
mpl.rcParams['pdf.fonttype'] = 3
# Output Type 3 (Type3) or Type 42 (TrueType), TrueType allows
# editing the text in illustrator


def figDiscretizeFS2D(self, kz = 0, meshXY = 1001,SaveFigure = False):
    """
    Show Discretized 2D Fermi Surface.
    """
    mesh_graph = meshXY
    kx = np.linspace(-pi / self.a, pi / self.a, mesh_graph)
    ky = np.linspace(-pi / self.b, pi / self.b, mesh_graph)
    kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

    fig, axes = plt.subplots(1, 1, figsize = (5.6, 5.6))
    fig.subplots_adjust(left = 0.24, right = 0.87, bottom = 0.29, top = 0.91)

    fig.text(0.39,0.84, r"$k_{\rm z}$ = 0", ha = "right", fontsize = 16)

    axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)

    nb_pkz = self.number_of_points_per_kz_list
    nkz = len(nb_pkz)
    # print("nkz : ", nkz)
    npkz0 = np.sum(nb_pkz[:nkz//2])
    # print("npkz0 : ", npkz0)
    npkz1 = npkz0 + nb_pkz[nkz//2]
    # print("npkz1 : ", npkz1)



    # line = axes.plot(self.kf[0, npkz0: npkz1] * self.a,
    #                  self.kf[1, npkz0: npkz1] * self.b)
    line = axes.plot(self.kf[0, :] * self.a,
                     self.kf[1, :] * self.b)
    # line = axes.plot(self.kf[0,:self.number_of_points_per_kz_list[0]] * self.a,
    #                  self.kf[1,:self.number_of_points_per_kz_list[0]] * self.b)
    plt.setp(line, ls ="", c = 'k', lw = 3, marker = "o", mfc = 'k', ms = 5, mec = "#7E2320", mew= 0)
    # axes.quiver(self.kf[0, npkz0: npkz1] * self.a,
    #             self.kf[1, npkz0: npkz1] * self.b,
    #             self.vf[0, npkz0: npkz1],
    #             self.vf[1, npkz0: npkz1],
    #             color = 'k')
    # modv = [np.sqrt(self.vf[0,j]**2 + self.vf[1,j]) for j in range(748)]
    
    axes.quiver(self.kf[0, :] * self.a,
                self.kf[1, :] * self.b,
                self.vf[0, :],
                self.vf[1, :],
                
                scale = 1e4,
                scale_units = "xy"
                )
    # axes.quiver(self.kf[0,:self.number_of_points_per_kz_list[0]] * self.a,
    #             self.kf[1,:self.number_of_points_per_kz_list[0]] * self.b,
    #             self.vf[0,:self.number_of_points_per_kz_list[0]],
    #             self.vf[1,:self.number_of_points_per_kz_list[0]],
    #             color = 'k')

    axes.set_xlim(-pi, pi)
    axes.set_ylim(-pi, pi)
    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
    axes.set_ylabel(r"$k_{\rm y}$", labelpad = 8)

    axes.set_xticks([-pi, 0., pi])
    axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    axes.set_yticks([-pi, 0., pi])
    axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

    
    if SaveFigure:
        plt.savefig("FermiSurface_quiver.pdf")

    plt.show()
def figDiscretizeFS3D(self, show_veloticites = False):
    """
    Show Discretized 3D Fermi Surface.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(self.kf[0,:], self.kf[1,:], self.kf[2,:], color='k', marker='.')
    if show_veloticites == True:
        ax.quiver(self.kf[0,:], self.kf[1,:], self.kf[2,:], self.vf[0,:], self.vf[1,:], self.vf[2,:], length=0.1, normalize=True)
    plt.show()

def figMultipleFS2D(self, meshXY = 1001, averaged_kz_FS = False):
    """
    Show 2D Fermi Surface for different kz.
    """
    mesh_graph = meshXY
    kx = np.linspace(-4*pi / self.a, 4*pi / self.a, mesh_graph)
    ky = np.linspace(-4*pi / self.b, 4*pi / self.b, mesh_graph)
    kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')

    fig, axes = plt.subplots(1, 1, figsize=(8.5, 5.6))
    fig.subplots_adjust(left=0.01, right=0.75, bottom=0.20, top=0.9)

    # doping_per_kz returns (n, p), so we look at p here.
    doping_per_kz = self.doping_per_kz(res_z=5)[1][2:]
    fig.text(0.63,0.84, r"$k_{\rm z}$ = 0,      $p$ $\in$ $k_{\rm z}$ = " + str(np.round(doping_per_kz[0], 3)), color = "#FF0000", fontsize = 18)
    fig.text(0.63,0.78, r"$k_{\rm z}$ = $\pi/c$,   $p$ $\in$ $k_{\rm z}$ = " + str(np.round(doping_per_kz[1], 3)), color = "#00DC39", fontsize = 18)
    fig.text(0.63,0.72, r"$k_{\rm z}$ = 2$\pi/c$, $p$ $\in$ $k_{\rm z}$ = " + str(np.round(doping_per_kz[2], 3)), color = "#6577FF", fontsize = 18)

    fig.text(0.63,0.3, r"Average over $k_{\rm z}$", fontsize = 18)
    fig.text(0.63,0.24, r"Total $p$ = " + str(np.round(self.doping(), 3)), fontsize = 18)

    axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 0), 0, colors = '#FF0000', linewidths = 3)
    axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, pi/self.c), 0, colors = '#00DC39', linewidths = 3)
    axes.contour(kxx*self.a, kyy*self.b, self.e_3D_func(kxx, kyy, 2*pi/self.c), 0, colors = '#6577FF', linewidths = 3)

    ## Averaged FS among all kz
    if averaged_kz_FS == True:
        kz_array = np.linspace(-2*pi/self.c, 2*pi/self.c, 5)
        dump = 0
        for kz in kz_array:
            dump += self.e_3D_func(kxx, kyy, kz)
        axes.contour(kxx*self.a, kyy*self.b, (1/self.res_z)*dump, 0, colors = '#000000', linewidths = 3, linestyles = "dashed")



    axes.set_xlim(-pi, pi)
    axes.set_ylim(-pi, pi)
    axes.tick_params(axis='x', which='major', pad=7)
    axes.tick_params(axis='y', which='major', pad=8)
    axes.set_xlabel(r"$k_{\rm x}$", labelpad = 8)
    axes.set_ylabel(r"$k_{\rm y}$", labelpad = -6)

    axes.set_xticks([-pi, 0., pi])
    axes.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    axes.set_yticks([-pi, 0., pi])
    axes.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    axes.set_aspect(aspect=1)

    plt.show()

def print_BZ(self,color = "k", alpha = 0.3):
    """
    Prints the Brillouin Zone in the axis ax.
    BZ must be a SpacegroupAnalizer.get_primitive_standard_structure.lattice.get_brillouin_zone() object
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    BZ = self.prim.lattice.get_brillouin_zone()
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