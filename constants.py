import numpy as np

# Antineutrino electron constants and fundamental constants
sin_theta_w = 0.22290 # https://physics.nist.gov/cgi-bin/cuu/Value?sin2th
E_mean = 12.28 # Tamborra, I., Müller, B., Hüdepohl, L., Janka, H. T., & Raffelt, G. (2012). High-resolution supernova neutrino spectra represented by a simple fit. Physical Review D, 86(12), 125031.
α = 2.61 # Tamborra, I., Müller, B., Hüdepohl, L., Janka, H. T., & Raffelt, G. (2012). High-resolution supernova neutrino spectra represented by a simple fit. Physical Review D, 86(12), 125031.
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