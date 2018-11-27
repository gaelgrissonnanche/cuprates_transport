import time
import numpy as np
from band import BandStructure
from admr import ADMR
from chambers import Conductivity
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

t = 533.616
bandObject = BandStructure(t=t, mu=-693.7/t, tp=-113.561/t,
                           tpp=23.2192/t, tz=8.7296719/t, tz2=-0.89335299/t, mesh_ds = np.pi / 28)

# bandObject.figDiscretizeFS2D()


# condObject = Conductivity(bandObject, Bamp=45, Bphi=0, Btheta=0, gamma_0 = 25*2*3.14, gamma_k = 0, power = 0)
# condObject.solveMovementFunc()
# condObject.figOnekft()

start_total_time = time.time()
ADMRObject = ADMR(bandObject, Bamp=45, gamma_0 = 25, gamma_k = 70, power = 12)
ADMRObject.runADMR()
print("ADMR time : %.6s seconds" % (time.time() - start_total_time))

ADMRObject.fileADMR()
ADMRObject.figADMR()



# mu_a = np.arange(0.81, 0.835, 0.001)
# for mu in mu_a:
#     mu = mu * t
#     print("mu = " + r"{0:.4f}".format(mu) + " * t" )
#     band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz, tz2], dtype = np.float64)
#     admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = False)

# gamma_k_a = np.arange(0, 120, 20)
# gamma_0_a = np.arange(50, 170, 20)
# for gamma_0 in gamma_0_a:
#     for gamma_k in gamma_k_a:
#         print("gamma_0 = " + r"{0:.1f}".format(gamma_0) + " THz")
#         print("gamma_k = " + r"{0:.1f}".format(gamma_k) + " THz")
#         tau_parameters = np.array([gamma_0, gamma_k, power], dtype = np.float64)
#         admrFunc(band_parameters, mesh_parameters, tau_parameters, B_amp, B_phi_a, B_theta_a, fig_show = False)



