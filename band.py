import numpy as np
from numpy import cos, sin, pi
from scipy import optimize
from skimage import measure
from numba import jit
from numba import jitclass          # import the decorator
from numba import int32, float32 

## Constant //////
hbar = 1 # velocity will be in units of 1 / hbar,
         # this hbar is taken into accound in the constant units_move_eq

## ABOUT JUST IN TIME (JIT) COMPILATION
# jitclass do not work, the best option is to call a jit otimized function from inside the class.

@jit(nopython = True, cache = True)
def optimized_e_3D_func(kx, ky, kz, a, b, c, mu, t, tp, tpp, tz, tz2):
    e_2D  = -2 * t * ( cos(kx*a) + cos(ky*b) ) 
    e_2D += -4 * tp * cos(kx*a) * cos(ky*b) 
    e_2D += -2 * tpp * ( cos(2*kx*a) + cos(2*ky*b) )
    e_2D += -mu

    e_z  = -2 * tz * cos(kz*c/2)
    e_z *= cos(kx*a/2) * cos(ky*b/2)
    e_z *= (cos(kx*a) - cos(ky*b) )**2
    # mind the '+='
    e_z += -2 * tz2 * cos(kz* c / 2.)

    return e_2D + e_z

@jit(nopython = True, cache = True)
def optimized_v_3D_func(kx, ky, kz, a, b, c,t, tp, tpp, tz, tz2):
    d = c / 2
    
    kxa = kx*a
    kxb = ky*b
    kzd = kz*d
    coskx = cos(kxa)
    cosky = cos(kxb)
    coskz = cos(kzd)
    sinkx = sin(kxa)
    sinky = sin(kxb)
    sinkz = sin(kzd)
    sin2kx = sin(2*kx*a)
    sin2ky = sin(2*ky*b)
    coskx_2 = cos(kxa/2)
    cosky_2 = cos(kxb/2)
    sinkx_2 = sin(kxa/2)
    sinky_2 = sin(kxb/2)

    d_e2D_dkx = 2*t*a*sinkx + 4*tp*a*sinkx*cosky + 4*tpp*a*sin2kx
    d_e2D_dky = 2*t*b*sinky + 4*tp*b*coskx*sinky + 4*tpp*b*sin2ky
    d_e2D_dkz = 0

    sigma = coskx_2*cosky_2
    d_sigma_dkx = - a/2*sinkx_2*cosky_2
    d_sigma_dky = - b/2*coskx_2*sinky_2
    diff = (coskx - cosky)
    square = (diff)**2
    d_square_dkx = 2*(diff)*(-a*sinkx)
    d_square_dky = 2*(diff)*(+b*sinky)

    d_ez_dkx = - 2*tz*d_sigma_dkx*square*coskz - 2*tz*sigma*d_square_dkx*coskz
    d_ez_dky = - 2*tz*d_sigma_dky*square*coskz - 2*tz*sigma*d_square_dky*coskz
    d_ez_dkz = - 2*tz*sigma*square*(-d*sinkz) + 2*tz2*(-d)*sinkz

    vx = (d_e2D_dkx + d_ez_dkx)/hbar
    vy = (d_e2D_dky + d_ez_dky)/hbar
    vz = (d_e2D_dkz + d_ez_dkz)/hbar

    return vx, vy, vz


def rotation(x, y, angle):
    xp =  cos(angle)*x + sin(angle)*y
    yp = -sin(angle)*x + cos(angle)*y
    return xp, yp

class BandStructure:
    def __init__(self):
        self.a   =  3.74
        self.b   =  3.74
        self.c   =  13.3
        self.t   =  190
        self.tp  = -0.14  *self.t
        self.tpp =  0.07  *self.t
        self.tz  =  0.07  *self.t
        self.tz2 =  0.00  *self.t
        self.mu  = -0.825 *self.t

        self.half_FS_z = True # if False, put a minimum of 11 points
        self.numberOfKz   = 7 
        self.mesh_ds = pi/20
        
        self.kf = None
        self.vf = None
        self.dkf = None
        self.numberPointsPerKz_list = None


    def bandParameters(self):
        return [self.a, self.b, self.c, self.mu, self.t, self.tp, self.tpp, self.tz, self.tz2]

    def e_3D_func(self,kx, ky, kz):
        return optimized_e_3D_func(kx, ky, kz, self.a, self.b, self.c, self.mu, self.t, self.tp, self.tpp, self.tz, self.tz2)
    
    def v_3D_func(self, kx, ky, kz):
        return optimized_v_3D_func(kx, ky, kz, self.a, self.b, self.c,self.t, self.tp, self.tpp, self.tz, self.tz2)
        
    def dispersionMesh(self, resX=500,resY=500, resZ=10):
        kx_a = np.linspace(-pi/self.a, pi/self.a, resX)
        ky_a = np.linspace(-pi/self.b, pi/self.b, resY)
        kz_a = np.linspace(-2*pi/self.c, 2*pi/self.c, resZ)
        kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a, indexing = 'ij')

        disp = self.e_3D_func(kxx, kyy, kzz)
        return disp

    def doping(self, resX=500, resY=500, resZ=10):
        disp = self.dispersionMesh(resX, resY, resZ)

        NumberOfkPoints = disp.shape[0] * disp.shape[1] * disp.shape[2]
        n = 2/NumberOfkPoints * np.sum( np.greater_equal(0, disp) )
        p = 1 - n
        
        return p
    
    def dopingPerkz(self, resX=500, resY=500, resZ=10):
        disp = self.dispersionMesh(resX, resY, resZ)
        
        NumberPerkz = disp.shape[0] * disp.shape[1]
        n_per_kz = 2 / NumberPerkz * np.sum( np.greater_equal(0, disp), axis = (0,1) )
        p_per_kz = 1 - n_per_kz

        return p_per_kz

    def dopingCondition(self, mu , ptarget ):
        self.mu = mu
        print(self.doping())
        return self.doping()-ptarget

    def setMuToDoping(self, pTarget, muStart = -8.0, xtol=0.001):
        solObject = optimize.root( self.dopingCondition, np.array([muStart]), args=(pTarget,),options={'xtol':xtol})
        self.mu = solObject.x[0]

    def discretize_FS(self):
        if self.numberOfKz % 2 == 0: # make it an odd number
            self.numberOfKz += 1  ## WARNING SIDE EFFECT ON MESH_Z

        mesh_xy_rough = 501 # make denser rough meshgrid to interpolate
        if self.half_FS_z: kz_a = np.linspace(0, 2*pi/self.c, self.numberOfKz) # half of FBZ, 2*pi/c because bodycentered unit cell
        else: kz_a = np.linspace(-2*pi/self.c, 2*pi/self.c, self.numberOfKz)
        kx_a = np.linspace(0, pi/self.a, mesh_xy_rough)
        ky_a = np.linspace(0, pi/self.b, mesh_xy_rough)
        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing = 'ij')

        self.numberPointsPerKz_list = []
        kxf_list = []
        kyf_list = []
        kzf_list = []
        dkf_list = []

        for j, kz in enumerate(kz_a):
            bands = self.e_3D_func(kxx, kyy, kz)
            contours = measure.find_contours(bands, 0)
            numberPointsPerKz = 0

            for i, contour in enumerate(contours):

                # Contour come in units proportionnal to size of meshgrid
                # one want to scale to units of kx and ky
                x = contour[:, 0]/(mesh_xy_rough-1)*pi
                y = contour[:, 1]/(mesh_xy_rough-1)*pi / (self.b/self.a) # anisotropy

                ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2) # segment lengths
                s = np.zeros_like(x) # arrays of zeros
                s[1:] = np.cumsum(ds) # integrate path, s[0] = 0

                mesh_xy = int( max(np.ceil(s.max() / self.mesh_ds), 4) )
                # choose at least a minimum of 4 points per contour
                numberPointsPerKz += mesh_xy

                dkf_weight = s.max() / (mesh_xy + 1) # weight to ponderate self.dkf

                s_int = np.linspace(0, s.max(), mesh_xy + 1) # regular spaced path, add one
                x_int = np.interp(s_int, s, x)[:-1] # interpolate and remove the last point (not to repeat)
                y_int = np.interp(s_int, s, y)[:-1]

                ## Rotate the contour to get the entire Fermi surface 
                # ### WARNING NOT ROBUST IN THE CASE OF C4 SYMMETRY BREAKING
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
                    kxf = x_int/self.a
                    kyf = y_int/self.a ## a becaus anisotropy was taken into account earlier
                    kzf = kz * np.ones_like(x_int)
                    self.dkf = dkf_weight * np.ones_like(x_int)
                else:
                    kxf = np.append(kxf, x_int/self.a)
                    kyf = np.append(kyf, y_int/self.a)  ## a becaus anisotropy was taken into account earlier
                    kzf = np.append(kzf, kz*np.ones_like(x_int))
                    self.dkf = np.append(self.dkf, dkf_weight * np.ones_like(x_int))

            self.numberPointsPerKz_list.append(4 * numberPointsPerKz)

        self.kf = np.vstack([kxf, kyf, kzf]) # dim -> (n, i0) = (xyz, position on FS)

        ## Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0,:], self.kf[1,:], self.kf[2,:])
        self.vf = np.vstack([vx, vy, vz]) # dim -> (i, i0) = (xyz, position on FS)

