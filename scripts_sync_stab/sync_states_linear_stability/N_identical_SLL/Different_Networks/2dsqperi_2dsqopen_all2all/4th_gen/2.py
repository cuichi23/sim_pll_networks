import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
delta = 84.37228318652858 * np.pi / 180
psi = 55.2217535040673 * np.pi / 180
n_S = 2.6726 + 3.0375j
phi_i = 70 * np.pi / 180
d_L = 300  # thickness of layer in nm
n_air = 1  # refractive index of air


def snell(phi, n1, n2):
    phi_ref = np.arcsin((n1 / n2) * np.sin(phi))
    return phi_ref


def fresnel(n1, phi1, n2, phi2):
    rs = (n1 * np.cos(phi1) - n2 * np.cos(phi2)) / (
        n1 * np.cos(phi1) + n2 * np.cos(phi2))
    rp = (n2 * np.cos(phi1) - n1 * np.cos(phi2)) / (
        n2 * np.cos(phi1) + n1 * np.cos(phi2))
    return rs, rp


def calc_rho(n_k):
    n = n_k[0]
    k = n_k[1]
    n_L = n + 1j * k
    phi_L = snell(phi_i, n_air, n_L)
    phi_S = snell(phi_L, n_L, n_S)
    rs_al, rp_al = fresnel(n_air, phi_i, n_L, phi_L)
    rs_ls, rp_ls = fresnel(n_L, phi_L, n_S, phi_S)
    beta = 2 * np.pi * d_L * n_L * np.cos(phi_L) / lambda_vac
    rp_L = (rp_al + rp_ls * np.exp(-2 * 1j * beta)) / (
        1 + rp_al * rp_ls * np.exp(-2 * 1j * beta))
    rs_L = (rs_al + rs_ls * np.exp(-2 * 1j * beta)) / (
        1 + rs_al * rs_ls * np.exp(-2 * 1j * beta))
    rho_L = rp_L / rs_L
    return abs(rho_L - rho_given)


lambda_vac = 300
n_S = 2.6726 + 3.0375j
rho_given = np.tan(psi) * np.exp(
    1j * delta)  # should be 1.4399295435287844+0.011780279522394433j
initialGuess = [1.5, 0.1]
nsolve = minimize(calc_rho, initialGuess)
print(nsolve)
