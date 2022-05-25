import numpy as np
import scipy.stats as st
from scipy.integrate import simps
import pandas as pd
from random import choices
from math import gamma
from constants import *

#Spectrum of neutrinos produced in the star
def produced_spectrum(x, specie, E_tot):
    if specie == 'nu_e':
        alpha = alpha_nu_e
        E_mean = E_mean_nu_e
    elif specie == 'nubar_e':
        alpha = alpha_nubar_e
        E_mean = E_mean_nubar_e
    elif specie == 'nu_x':
        alpha = alpha_nu_x
        E_mean = E_mean_nu_x
    else:
        raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')
    L = E_tot*6.2415e5
    A = L*((1+alpha)**(1+alpha))/((gamma(1+alpha))*E_mean**(2+alpha))
    f_nu = A*(x**alpha)*np.exp(-(alpha + 1)*x/E_mean)
    return f_nu

#Final spectrum at the surface due to oscillation in stellar medium and crossing earth
"""
1. Dighe, A. S., & Smirnov, A. Y. (2000). Identifying the neutrino mass
spectrum from a supernova neutrino burst. Physical Review D, 62(3), 033007.

2. Agafonova, N. Y., Aglietta, M., Antonioli, P., Bari, G., Boyarkin, V. V., Bruno,
G., ... & Zichichi, A. (2007). Study of the effect of neutrino oscillations on the
supernova neutrino signal in the LVD detector. Astroparticle Physics, 27(4), 254-270.

3. Brdar, V., & Xu, X. J. (2022). Timing and Multi-Channel: Novel Method for
Determining the Neutrino Mass Ordering from Supernovae. arXiv preprint arXiv:2204.13135.

4. Dighe, A. S., Kachelriess, M., Raffelt, G. G., & Tomas, R. (2004). Signatures of
supernova neutrino oscillations in the earth mantle and core. Journal of Cosmology and
Astroparticle Physics, 2004(01), 004.

5. Borriello, E., Chakraborty, S., Mirizzi, A., Serpico, P. D., & Tamborra, I. (2012). Can
one observe Earth matter effects with supernova neutrinos?. Physical Review D, 86(8), 083004.
"""

def theta_matter(x, N_e):
    """
    Effective neutrino mixing angle in matter
    see equation (2) above
    """
    A = 2*np.sqrt(2)*G_F*x*N_e
    thetam = np.arcsin(np.sin(2*theta_12)/np.sqrt((A/delta_m['m_10'] - np.cos(2*theta_12))**2 + np.sin(2*theta_12)**2))/2
    return thetam

def earth_oscillation(x, phi):
    """
    Oscillatory contribution to energy spectrum due to Earth crossing by neutrinos.
    See references 4 and 5 for further detail.
    """
    y = 12.5/x
    L_tot = R_earth*np.sqrt(2*(1 - np.cos(2*phi))) # km
    if phi > 0.9947:
        L_m = (R_earth*np.sin(phi) - np.sqrt(R_core**2 - (np.cos(phi)**2)*(R_earth**2)))
    else:
        L_m = L_tot
    L_c = L_tot - L_m
    theta_m = theta_matter(x, N_e_m)
    theta_c = theta_matter(x, N_e_c)
    
    As = [-0.5*np.sin(2*theta_12 - 4*theta_m)*np.sin(4*theta_c - 4*theta_m),
          (np.cos(theta_c - theta_m)**2)*np.sin(2*theta_12 - 4*theta_m)*np.sin(2*theta_c - 2*theta_m),
          np.sin(2*theta_12 - 2*theta_m)*(np.cos(theta_c - theta_m)**4)*np.sin(2*theta_m),
          -(np.sin(2*theta_c - 2*theta_m)**2)*(np.cos(2*theta_12 - 4*theta_m) - 0.5*np.sin(2*theta_12 - 2*theta_m)*np.sin(2*theta_m)),
          0.5*np.sin(2*theta_12 - 2*theta_m)*np.sin(2*theta_m)*np.sin(2*theta_c - 2*theta_m)**2,
          -2*np.sin(2*theta_12 - 4*theta_m)*np.cos(theta_c - theta_m)*np.sin(theta_c - theta_m)**3,
          np.sin(2*theta_12 - 2*theta_m)*np.sin(2*theta_m)*np.sin(theta_c - theta_m)**4]
    
    sigma_m = 2*delta_m['m_10']*np.sin(2*theta_12)*L_m*y/(np.sin(2*theta_m)*1e3*1e-5)
    sigma_c = 2*delta_m['m_10']*np.sin(2*theta_12)*L_c*y/(np.sin(2*theta_c)*1e3*1e-5)
    sigmas = [sigma_m/2, sigma_m/2 + sigma_c, sigma_m + sigma_c, sigma_c, sigma_m,
              sigma_m/2 - sigma_c, sigma_m - sigma_c]
    
    P = 0
    for i in range(7):
        P += As[i]*np.sin(sigmas[i]/2)**2

    return P

def emitted_spectrum(x, flavor, E_tot, hierarchy = 'normal', phi = 0):
    F_e = produced_spectrum(x, 'nu_e', E_tot)
    F_ebar = produced_spectrum(x, 'nubar_e', E_tot)
    F_x = produced_spectrum(x, 'nu_x', E_tot)
    if hierarchy == 'normal':
        a_e = s_12**2
        b_e = 0
        c_e = c_12**2
        a_ebar = 0
        b_ebar = 0
        c_ebar = 1
        a_x = 0.25*(2 + s_12**2)
        b_x = 0.25
        c_x = 0.25*c_12**2
    elif hierarchy == 'inverted':
        a_e = 0
        b_e = 0
        c_e = 1
        a_ebar = 0
        b_ebar = c_12**2
        c_ebar = s_12**2
        a_x = 0.25*(2 + c_12**2)
        b_x = 0.25
        c_x = 0.25*s_12**2
    else:
        raise ValueError('Invalid type of hierarchy, please use "normal" or "inverted"')
    if flavor == 'nu_e':
        return a_e*F_e + b_e*F_ebar + c_e*F_x
    elif flavor == 'nubar_e':
        return a_ebar*F_e + b_ebar*F_ebar + c_ebar*F_x + (F_ebar - F_x)*earth_oscillation(x, phi)
    elif flavor == 'nu_x':
        return a_x*F_e + b_x*F_ebar + c_x*F_x
    else:
        raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')