import numpy as np

# Antineutrino electron constants and fundamental constants
sin_theta_w = 0.22290 # https://physics.nist.gov/cgi-bin/cuu/Value?sin2th
params = {'Parameters': ['alpha_nu_e','alpha_nubar_e','alpha_nu_x',
                        'E_mean_nu_e','E_mean_nubar_e','E_mean_nu_x',
                        'L_nu_e', 'L_nubar_e', 'L_nu_x'],
        'LS220-11.2': [2.96,2.55,2.12,
                      10.31,12.90,12.86,
                      3.5e52,3.09e52,4*3.01e52],
        'LS220-27.0': [2.76,2.47,2.07,
                      11.06,13.82,13.76,
                      5.7e52,5.4e52,4*5.06e52],
        'Shen-11.2': [2.95,2.37,2.00,
                     9.77,12.01,11.65,
                     3.21e52,2.83e52,4*2.54e52],
        'Shen-27.0': [2.72,2.25,1.92,
                     10.31,12.67,12.17,
                     5.27e52,5.00e52,4*4.27e52]} # Garching group simulations
g1_barnu_e = sin_theta_w
g2_barnu_e = 1/2 + sin_theta_w
sigma_0 = 88.06e-46 #cm^2
# Other neutrinos
g1_nu_e = 1/2 + sin_theta_w
g2_nu_e = sin_theta_w
g1_nu_x = -1/2 + sin_theta_w
g2_nu_x = sin_theta_w
g1_barnu_x = sin_theta_w
g2_barnu_x = -1/2 + sin_theta_w

#Neutrino mixing matrix (PMNS matrix)
theta_12 = 33.447*np.pi/180
theta_23 = 45*np.pi/180
theta_13 = 8.878*np.pi/180
delta_cp = 0#195*2*np.pi/360

c_12 = np.cos(theta_12)
c_13 = np.cos(theta_13)
c_23 = np.cos(theta_23)
s_12 = np.sin(theta_12)
s_13 = np.sin(theta_13)
s_23 = np.sin(theta_23)

U = np.array([[c_12*c_13, s_12*c_13, s_13*np.exp(-1j*delta_cp)],
    [-s_12*c_23-c_12*s_23*s_13*np.exp(1j*delta_cp), c_12*c_23-s_12*s_23*s_13*np.exp(1j*delta_cp), s_23*c_13],
    [s_12*s_23-c_12*c_23*s_13*np.exp(1j*delta_cp), -c_12*s_23-s_12*c_23*s_13*np.exp(1j*delta_cp), c_23*c_13]])

U23 = np.array([ [1,0,0],[0,c_23,s_23],[0,-s_23,c_23] ])
U13 = np.array([ [c_13,0,s_13],[0,1,0],[-s_13,0,c_13] ])
U12 = np.array([ [c_12,s_12,0],[-s_12,c_12,0],[0,0,1] ])
U3 = U23 @ U13 @ U12
U3_dag = np.transpose(U3)
# For reference in the terms, the matrix is organized in the following way
"""
    | Ue1 Ue2 Ue3 |
U = | Uμ1 Uμ2 Uμ3 |
    | Uτ1 Uτ2 Uτ3 |
"""

delta_m = {'m_10': 7.53e-5, 'm_20': 2.44e-3, 'm_21': 2.44e-3,
           'm_01': 7.53e-5, 'm_02': 2.44e-3, 'm_12': 2.44e-3}

G_F = 1.16e-5 # Fermi constant in GeV⁻²
Gf = 1.16632e-23 # Fermi constant in eV⁻²
Grav_const = 6.674e-11 # Universal gravitational constant
""""
Bernabeu, J., Palomares-Ruiz, S., Perez, A., & Petcov, S. T. (2002). The earth
Mantle-Core effect in charge-asymmetries for atmospheric neutrino oscillations.
Physics Letters B, 531(1-2), 90-98.
"""
N_e_m = -150060250216.29025 # Electron number density in the mantle
N_e_c = -391697925762.5642 # Electron number density in the core


# CKM matrix
V_ckm = np.array([[0.97370, 0.2245, 0.00382],
              [0.221, 0.987, 0.0410],
              [0.0080, 0.0388, 1.013]])
# For reference in the terms, the matrix is organized in the following way
"""
    | Vud Vus Vub |
U = | Vcd Vcs Vcb |
    | Vtd Vts Vtb |
"""

# Detection thresholds for detectors
E_th = {'super-k': 4.5,
        'Hyper-k': 4.5,
        'DUNE': 3,
        'JUNO': 0.3}

# Earth information
R_earth = 6371 # Km
R_core = 3486 # Km
Avog = 6.022e23 # Avogadro number
c = 2.99792458e8 # m/s
cm3_to_eV3 = (0.197e9*1e-15*100)**3 # Transformation from cm³ to eV³

detectors_lat = {'super-k': 36.42*np.pi/180,
                 'Hyper-k': 36.42*np.pi/180,
                 'DUNE': 41.817*np.pi/180,
                 'JUNO': 22.372*np.pi/180}
detectors_lon = {'super-k': 137.31*np.pi/180,
                 'Hyper-k': 137.31*np.pi/180,
                 'DUNE': -88.257*np.pi/180,
                 'JUNO': 112.57*np.pi/180}