import numpy as np

# Antineutrino electron constants and fundamental constants
sin_theta_w = 0.22290 # https://physics.nist.gov/cgi-bin/cuu/Value?sin2th
E_mean_nubar_e = 12.28 # Tamborra, I., Müller, B., Hüdepohl, L., Janka, H. T., & Raffelt, G. (2012). High-resolution supernova neutrino spectra represented by a simple fit. Physical Review D, 86(12), 125031.
alpha_nubar_e = 2.61 # Tamborra, I., Müller, B., Hüdepohl, L., Janka, H. T., & Raffelt, G. (2012). High-resolution supernova neutrino spectra represented by a simple fit. Physical Review D, 86(12), 125031.
E_mean_nu_e = 10.01 # Tamborra, I., Müller, B., Hüdepohl, L., Janka, H. T., & Raffelt, G. (2012). High-resolution supernova neutrino spectra represented by a simple fit. Physical Review D, 86(12), 125031.
alpha_nu_e = 2.92 # Tamborra, I., Müller, B., Hüdepohl, L., Janka, H. T., & Raffelt, G. (2012). High-resolution supernova neutrino spectra represented by a simple fit. Physical Review D, 86(12), 125031.
E_mean_nu_x = 12.31 # Tamborra, I., Müller, B., Hüdepohl, L., Janka, H. T., & Raffelt, G. (2012). High-resolution supernova neutrino spectra represented by a simple fit. Physical Review D, 86(12), 125031.
alpha_nu_x = 2.34 # Tamborra, I., Müller, B., Hüdepohl, L., Janka, H. T., & Raffelt, G. (2012). High-resolution supernova neutrino spectra represented by a simple fit. Physical Review D, 86(12), 125031.
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
theta_12 = 33.447*2*np.pi/360
theta_23 = 45*2*np.pi/360
theta_13 = 8.878*2*np.pi/360
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
# For reference in the terms, the matrix is organized in the following way
"""
    | Ue1 Ue2 Ue3 |
U = | Uμ1 Uμ2 Uμ3 |
    | Uτ1 Uτ2 Uτ3 |
"""

delta_m = {'m_10': 7.53e-5, 'm_20': 2.44e-3, 'm_21': 2.44e-3,
           'm_01': 7.53e-5, 'm_02': 2.44e-3, 'm_12': 2.44e-3}