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
def optimized_v_2D_func(kx, ky, a, b, t, tp, tpp):
    d_e2D_dkx =  2 * t * a * sin(kx*a) 
    d_e2D_dkx += 4 * tp * a * sin(kx*a)*cos(ky*b)
    d_e2D_dkx += 4 * tpp * a * sin(2*kx*a)
    
    d_e2D_dky =  2 * t * b * sin(ky*b) 
    d_e2D_dky += 4 * tp * b * cos(kx*a)*sin(ky*b)
    d_e2D_dky += 4 * tpp * b * sin(2*ky*b)

    return d_e2D_dkx, d_e2D_dky

@jit(nopython = True, cache = True)
def optimized_v_ez_func(kx, ky, kz, a, b, c, tz, tz2):
    sigma = cos(kx*a/2) * cos(ky*b/2)
    d_sigma_dkx = - a / 2 * sin(kx*a/2) * cos(ky*b/2)
    d_sigma_dky = - b / 2 * cos(kx*a/2) * sin(ky*b/2)

    square = (cos(kx*a) - cos(ky*b))**2
    d_square_dkx = 2 * square * (-a * sin(kx*a))
    d_square_dky = 2 * square * (+b * sin(ky*b))
    
    d_ez_dkx =  -2 * tz * d_sigma_dkx * square * cos(kz*c/2) 
    d_ez_dkx += -2 * tz * sigma * d_square_dkx * cos(kz*c/2)
    
    d_ez_dky =  -2 * tz * d_sigma_dky * square * cos(kz*c/2) 
    d_ez_dky += -2 * tz * sigma * d_square_dky * cos(kz*c/2)

    d_ez_dkz =  -2 * tz * sigma * square * (-c/2 * sin(kz*c/2)) 
    d_ez_dkz += -2 * tz2 * (-c/2) * sin(kz*c/2)
    
    return d_ez_dkx, d_ez_dky, d_ez_dkz

class BandStructure:
    def __init__(self):
        self.a   =  3.74
        self.b   =  3.74
        self.c   =  13.3
        self.t   =  190
        self.tp  = -0.14 *self.t
        self.tpp =  0.07 *self.t
        self.tz  =  0.07 *self.t
        self.tz2 =  0.00 *self.t
        self.mu  = -0.81 *self.t

        self.half_FS_z = True # if False, put a minimum of 11 points
        self.numberOfKz   = 7 
        self.numberOfKxKy = 56
        
        self.kf = None
        self.vf = None
        self.dky = None
        self.number_contours = None


    def bandParameters(self):
        return [self.a, self.b, self.c, self.mu, self.t, self.tp, self.tpp, self.tz, self.tz2]

    def e_3D_func(self,kx, ky, kz):
        return optimized_e_3D_func(kx, ky, kz, self.a, self.b, self.c, self.mu, self.t, self.tp, self.tpp, self.tz, self.tz2)
    

    def v_2D_func(self, kx, ky):
        return optimized_v_2D_func(kx, ky, self.a, self.b, self.t, self.tp, self.tpp)

    def v_ez_func(self, kx, ky, kz):
        return optimized_v_ez_func(kx, ky, kz, self.a, self.b, self.c, self.tz, self.tz2)

    def v_3D_func(self, kx, ky, kz):
        d_e2D_dkx, d_e2D_dky = self.v_2D_func(kx, ky)
        d_ez_dkx, d_ez_dky, d_ez_dkz = self.v_ez_func(kx, ky, kz)

        vx = ( 1/hbar ) * (d_e2D_dkx + d_ez_dkx)
        vy = ( 1/hbar ) * (d_e2D_dky + d_ez_dky)
        vz = ( 1/hbar ) * (d_ez_dkz)

        return vx, vy, vz
        
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
        self.numberOfKxKy -= (self.numberOfKxKy % 4) #ensures fourfold symmetry of sampling
        
        mesh_xy_rough = self.numberOfKxKy * 10 + 1 # make denser rough meshgrid to interpolate
        kx_a = np.linspace(-pi/self.a, pi/self.a, mesh_xy_rough)
        ky_a = np.linspace(-pi/self.b, pi/self.b, mesh_xy_rough)
        kxx, kyy = np.meshgrid(kx_a, ky_a, indexing = 'ij')

        kz_a = np.linspace(-2*pi/self.c, 2*pi/self.c, self.numberOfKz) # 2*pi/c because bodycentered unit cell
        if self.half_FS_z == True: kz_a = np.linspace(0, 2*pi/self.c, self.numberOfKz)
        
        for j, kz in enumerate(kz_a):
            bands = self.e_3D_func(kxx, kyy, kz)
            contours = measure.find_contours(bands, 0)
            self.number_contours = len(contours)

            for i, contour in enumerate(contours):

                # Contour in units proportionnal to size of meshgrid
                x_raw = contour[:, 0]
                y_raw = contour[:, 1]

                # Scale the contour to units of kx and ky
                x = (x_raw/(mesh_xy_rough-1)-0.5)*2*pi/self.a
                y = (y_raw/(mesh_xy_rough-1)-0.5)*2*pi/self.b


                # Is Contour closed?
                closed = (x_raw[0] == x_raw[-1]) * (y_raw[0] == y_raw[-1])

                # Make all opened contours go in direction of x[i+1] - x[i] < 0, otherwise interpolation problem
                if np.diff(x)[0] > 0 and not closed:
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
                    self.numberOfKxKy = self.numberOfKxKy - (self.numberOfKxKy % 4) # respects the 4-order symmetry

                    ds = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
                    s = np.zeros_like(x) # arrays of zeros
                    s[1:] = np.cumsum(ds) # integrate path, s[0] = 0

                    s_int = np.linspace(0, s.max(), self.numberOfKxKy + 1) # regular spaced path
                    x_int = np.interp(s_int, s, x)[:-1] # interpolate
                    y_int = np.interp(s_int, s, y)[:-1]

                # Opened contour
                else:
                    ds = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
                    s = np.zeros_like(x) # arrays of zeros
                    s[1:] = np.cumsum(ds) # integrate path, s[0] = 0

                    s_int = np.linspace(0, s.max(), self.numberOfKxKy) # regular spaced path
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

        self.kf = np.vstack([kxf, kyf, kzf]) # dim -> (n, i0) = (xyz, position on FS)

        ## Integration Delta (WROOOOONNNGGGG!!!!)
        if self.half_FS_z == True:
            self.dkf = 2 / (self.numberOfKxKy * self.numberOfKz) * ( 2 * pi )**2 / ( self.a * self.b ) * ( 4 * pi ) / self.c
        else:
            self.dkf = 1 / (self.numberOfKxKy * self.numberOfKz) * ( 2 * pi )**2 / ( self.a * self.b ) * ( 4 * pi ) / self.c

        ## Compute Velocity at t = 0 on Fermi Surface
        vx, vy, vz = self.v_3D_func(self.kf[0,:], self.kf[1,:], self.kf[2,:])
        self.vf = np.vstack([vx, vy, vz]) # dim -> (i, i0) = (xyz, position on FS)























