import numpy as np
from numpy import cos, sin, pi, sqrt, arctan2, cosh, arccosh
from scipy.constants import hbar

from cuprates_transport.utils import meV, Angstrom, kB
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Scatterint rate models ---------------------------------------------------#
# I think we might as well want to create a Gamma object.

def gamma_DOS_func(condObj, vx, vy, vz):
    dos = 1 / sqrt( vx**2 + vy**2 + vz**2 )
    dos_max = np.max(condObj.bandObject.dos_k)
    # value to normalize the DOS to a quantity without units
    return condObj.gamma_dos_max * (dos / dos_max)

def gamma_cmfp_func(condObj, vx, vy, vz):
    vf = sqrt( vx**2 + vy**2 + vz**2 )
    vf_max = np.max(condObj.bandObject.vf)
    # value to normalize the DOS to a quantity without units
    return condObj.gamma_cmfp_max * (vf / vf_max)

def gamma_vF_func(condObj, vx, vy, vz):
    """vx, vy, vz are in Angstrom.meV
    l_path is in Angstrom
    """
    vF = sqrt(vx**2 + vy**2 + vz**2) / Angstrom * 1e-12 # in Angstrom / ps
    return vF / condObj.l_path

def gamma_k_func(condObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = condObj.bandObject.a, condObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx) #+ np.pi/4
    return condObj.gamma_k * np.abs(cos(2*phi))**condObj.power

def gamma_coskpi4_func(condObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = condObj.bandObject.a, condObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx) #+ np.pi/4
    return condObj.gamma_kpi4 * np.abs(cos(2*(phi+1*pi/4)))**condObj.powerpi4

def gamma_poly_func(condObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = condObj.bandObject.a, condObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx)
    phi_p = np.abs((np.mod(phi, pi/2)-pi/4))
    return (condObj.a0 + np.abs(condObj.a1 * phi_p + condObj.a2 * phi_p**2 +
            condObj.a3 * phi_p**3 + condObj.a4 * phi_p**4 + condObj.a5 * phi_p**5))

def gamma_sinn_cosm(condObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = condObj.bandObject.a, condObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx) #+ np.pi/4
    return condObj.a0 + condObj.a1 * np.abs(cos(2*phi))**condObj.a2 + condObj.a3 * np.abs(sin(2*phi))**condObj.a4

def gamma_tanh_func(condObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = condObj.bandObject.a, condObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx)
    return condObj.a0 / np.abs(np.tanh(condObj.a1 + condObj.a2 * np.abs(cos(2*(phi+pi/4)))**condObj.a3))

def gamma_step_func(condObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = condObj.bandObject.a, condObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx)
    index_low = ((np.mod(phi, pi/2) >= (pi/4 - condObj.phi_step)) *
                (np.mod(phi, pi/2) <= (pi/4 + condObj.phi_step)))
    index_high = np.logical_not(index_low)
    gamma_step_array = np.zeros_like(phi)
    gamma_step_array[index_high] = condObj.gamma_step
    return gamma_step_array

def gamma_ndlsco_tl2201_func(condObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = condObj.bandObject.a, condObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx)
    return (condObj.a0 + condObj.a1 * np.abs(cos(2*phi))**12 +
            condObj.a2 * np.abs(cos(2*phi))**2)

def gamma_skew_marginal_fl(condObj, epsilon):
    return np.sqrt((condObj.a_epsilon * epsilon +
                    condObj.a_abs_epsilon * np.abs(epsilon))**2 +
                    (condObj.a_T * kB * meV / hbar * 1e-12 * condObj.T)**2)

def gamma_fl(condObj, epsilon):
    return (condObj.a_epsilon_2 * epsilon**2 +
            condObj.a_T2 * (kB * meV / hbar * 1e-12 * condObj.T)**2)

def gamma_skew_planckian(condObj, epsilon):
    x = epsilon / (kB * condObj.T)
    x = np.where(x == 0, 1.0e-20, x)
    return ((condObj.a_asym * kB * condObj.T) * ((x + condObj.p_asym)/2) * np.cosh(x/2) /
            np.sinh((x + condObj.p_asym)/2) / np.cosh(condObj.p_asym/2))
