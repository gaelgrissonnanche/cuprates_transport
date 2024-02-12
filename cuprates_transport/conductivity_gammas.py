import numpy as np
from numpy import cos, sin, pi, sqrt, arctan2
from scipy.constants import Boltzmann, hbar, elementary_charge, \
    physical_constants
##<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

## Units ////////
meV = physical_constants["electron volt"][0] * 1e-3 # 1 meV in Joule
Angstrom = 1e-10 # 1 A in meters
picosecond = 1e-12 # 1 ps in seconds

## Constant //////
hbar = hbar # m2 kg / s
e = elementary_charge # C
kB = Boltzmann # J / K
kB = kB / meV # meV / K

## This coefficient takes into accound all units and constant to prefactor the movement equation
units_move_eq =  e * Angstrom**2 * picosecond * meV / hbar**2

## This coefficient takes into accound all units and constant to prefactor Chambers formula
units_chambers = 2 * e**2 / (2*pi)**3 * meV * picosecond / Angstrom / hbar**2


def gamma_DOS_func(cObj, vx, vy, vz):
    dos = 1 / sqrt( vx**2 + vy**2 + vz**2 )
    dos_max = np.max(cObj.bandObject.dos_k)
    # value to normalize the DOS to a quantity without units
    return cObj.gamma_dos_max * (dos / dos_max)

def gamma_cmfp_func(cObj, vx, vy, vz):
    vf = sqrt( vx**2 + vy**2 + vz**2 )
    vf_max = np.max(cObj.bandObject.vf)
    # value to normalize the DOS to a quantity without units
    return cObj.gamma_cmfp_max * (vf / vf_max)

def gamma_vF_func(cObj, vx, vy, vz):
    """vx, vy, vz are in Angstrom.meV
       l_path is in Angstrom
    """
    vF = sqrt(vx**2 + vy**2 + vz**2) / (hbar / meV) * 1e-12 # in Angstrom / ps
    return vF / cObj.l_path

def gamma_k_func(cObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = cObj.bandObject.a, cObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx) #+ np.pi/4
    return cObj.gamma_k * np.abs(cos(2*phi))**cObj.power

def gamma_coskpi4_func(cObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = cObj.bandObject.a, cObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx) #+ np.pi/4
    return cObj.gamma_kpi4 * np.abs(cos(2*(phi+1*pi/4)))**cObj.powerpi4

def gamma_poly_func(cObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = cObj.bandObject.a, cObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx)
    phi_p = np.abs((np.mod(phi, pi/2)-pi/4))
    return (cObj.a0 + np.abs(cObj.a1 * phi_p + cObj.a2 * phi_p**2 +
            cObj.a3 * phi_p**3 + cObj.a4 * phi_p**4 + cObj.a5 * phi_p**5))

def gamma_sinn_cosm(cObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = cObj.bandObject.a, cObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx) #+ np.pi/4
    return cObj.a0 + cObj.a1 * np.abs(cos(2*phi))**cObj.a2 + cObj.a3 * np.abs(sin(2*phi))**cObj.a4

def gamma_tanh_func(cObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = cObj.bandObject.a, cObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx)
    return cObj.a0 / np.abs(np.tanh(cObj.a1 + cObj.a2 * np.abs(cos(2*(phi+pi/4)))**cObj.a3))

def gamma_step_func(cObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = cObj.bandObject.a, cObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx)
    index_low = ((np.mod(phi, pi/2) >= (pi/4 - cObj.phi_step)) *
                 (np.mod(phi, pi/2) <= (pi/4 + cObj.phi_step)))
    index_high = np.logical_not(index_low)
    gamma_step_array = np.zeros_like(phi)
    gamma_step_array[index_high] = cObj.gamma_step
    return gamma_step_array

def gamma_ndlsco_tl2201_func(cObj, kx, ky, kz):
    ## Make sure kx and ky are in the FBZ to compute Phi.
    a, b = cObj.bandObject.a, cObj.bandObject.b
    kx = np.remainder(kx + pi / a, 2*pi / a) - pi / a
    ky = np.remainder(ky + pi / b, 2*pi / b) - pi / b
    phi = arctan2(ky, kx)
    return (cObj.a0 + cObj.a1 * np.abs(cos(2*phi))**12 +
            cObj.a2 * np.abs(cos(2*phi))**2)

def gamma_skew_marginal_fl(cObj, epsilon):
    return np.sqrt((cObj.a_epsilon * epsilon +
                    cObj.a_abs_epsilon * np.abs(epsilon))**2 +
                    (cObj.a_T * kB * meV / hbar * 1e-12 * cObj.T)**2)

def gamma_fl(cObj, epsilon):
    return (cObj.a_epsilon_2 * epsilon**2 +
            cObj.a_T2 * (kB * meV / hbar * 1e-12 * cObj.T)**2)

def gamma_skew_planckian(cObj, epsilon):
    x = epsilon / (kB * cObj.T)
    x = np.where(x == 0, 1.0e-20, x)
    return ((cObj.a_asym * kB * cObj.T) * ((x + cObj.p_asym)/2) * np.cosh(x/2) /
            np.sinh((x + cObj.p_asym)/2) / np.cosh(cObj.p_asym/2))
