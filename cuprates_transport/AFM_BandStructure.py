## Antiferromagnetic Reconstruction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
import sympy as sp
from cuprates_transport.bandstructure import BandStructure
from numba import jit

class AFMBandStructure(BandStructure):
    def __init__(self, electronPocket=False, reconstruction_3D=False, Q_vector=1, **kwargs):
        super().__init__(**kwargs)
        self._electronPocket    = electronPocket
        self._reconstruction_3D = reconstruction_3D # if True the reconstruction is over E_2D + E_z, otherwise just E_2D
        self._Q_vector          = Q_vector # if = 1 then Q=(pi,pi), if =-1 then Q=(-pi,pi), it only matters if reconstruction_3D = True
        self.numberOfBZ         = 2  # number of BZ we intregrate on as we still work on the unreconstructed FBZ

        ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        try:
            assert self._band_params["M"] > 0.00001
        except KeyError:
            self._band_params["M"] = 0.00001
            print("Warning! 'M' has to be defined; it has been added and set to 0.00001")
        except AssertionError:
            self._band_params["M"] = 0.00001
            print("Warning! 'M' has to be > 0.00001; it has been set to 0.00001")

        ## Build the symbolic variables
        self.var_sym = [sp.Symbol('kx'), sp.Symbol('ky'), sp.Symbol('kz'),
                        sp.Symbol('a'),  sp.Symbol('b'),  sp.Symbol('c')]
        for params in sorted(self._band_params.keys()):
            self.var_sym.append(sp.Symbol(params))
        self.var_sym = tuple(self.var_sym)

        ## Create the dispersion and velocity functions
        self.e_3D_v_3D_AFM_definition()

    ## Properties >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def _get_electronPocket(self):
        return self._electronPocket
    def _set_electronPocket(self, electronPocket):
        print("You can only set this parameter when building the object")
    electronPocket = property(_get_electronPocket, _set_electronPocket)

    def _get_reconstruction_3D(self):
        return self._reconstruction_3D
    def _set_reconstruction_3D(self, reconstruction_3D):
        print("You can only set this parameter when building the object")
    reconstruction_3D = property(_get_reconstruction_3D, _set_reconstruction_3D)

    def _get_Q_vector(self):
        return self._Q_vector
    def _set_Q_vector(self, Q_vector):
        print("You can only set this parameter when building the object")
    Q_vector = property(_get_Q_vector, _set_Q_vector)

    ## Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    def e_3D_v_3D_AFM_definition(self):

        """Defines with Sympy the dispersion relation and
        symbolicly derives the velocity"""

        ## Symbolic variables ///////////////////////////////////////////////////
        kx = sp.Symbol('kx')
        ky = sp.Symbol('ky')
        kz = sp.Symbol('kz')
        a  = sp.Symbol('a')
        b  = sp.Symbol('b')
        mu = sp.Symbol('mu')
        M  = sp.Symbol('M')

        ## Dispersion //////////////////////////////////////////////////////////
        if self._electronPocket == True:
            print("Electron pocket")
            sign_pocket = 1
        else:
            print("Hole pocket")
            sign_pocket = -1

        # if self._reconstruction_3D == True:
        #     self.epsilon_sym = self.epsilon_xy_sym + self.epsilon_z_sym
        # else:
        #     self.epsilon_sym = self.epsilon_xy_sym
        self.epsilon_sym = self.energy_sym

        self.epsilon_AFM_sym = 0.5 * (self.epsilon_sym + self.epsilon_sym.subs([(kx, kx+ self._Q_vector*sp.pi/a), (ky, ky+sp.pi/b)])) + \
            sign_pocket * sp.sqrt(0.25*(self.epsilon_sym - self.epsilon_sym.subs([(kx, kx+ self._Q_vector*sp.pi/a), (ky, ky+sp.pi/b)]))**2 + M**2)


        ## Velocity ////////////////////////////////////////////////////////////
        self.v_PiPi_sym = [sp.diff(self.epsilon_PiPi_sym, kx), 
                           sp.diff(self.epsilon_PiPi_sym, ky), 
                           sp.diff(self.epsilon_PiPi_sym, kz)]
        # Check is one of the velocitiy components is "0" ////////////////////
        k_list = ['kx', 'ky', 'kz']
        for i, v in enumerate(self.v_PiPi_sym):
            if v == 0:
                self.v_PiPi_sym[i] = "numpy.zeros_like(" + k_list[i] + ")"
        ## Lambdafity //////////////////////////////////////////////////////////
        epsilon_func = sp.lambdify(self.var_sym, self.epsilon_PiPi_sym, 'numpy')
        v_func = sp.lambdify(self.var_sym, self.v_PiPi_sym, 'numpy')

        ## Numba ////////////////////////////////////////////////////////////////
        # self.epsilon_func = jit(epsilon_func, nopython=True)
        # self.v_func = jit(v_func, nopython=True)
        if self.parallel is True:
            self.epsilon_func = jit(epsilon_func, nopython=True, parallel=True)
            self.v_func = jit(v_func, nopython=True, parallel=True)
        else:
            self.epsilon_func = jit(epsilon_func, nopython=True)
            self.v_func = jit(v_func, nopython=True)
