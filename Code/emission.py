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
    A = E_tot*((1+alpha)**(1+alpha))/((gamma(1+alpha))*E_mean**(2+alpha))
    f_nu = A*(x**alpha)*np.exp(-(alpha + 1)*x/E_mean)
    return f_nu

#Final spectrum at the surface due to oscillation in stellar medium
"""
1. Dighe, A. S., & Smirnov, A. Y. (2000). Identifying the neutrino mass
spectrum from a supernova neutrino burst. Physical Review D, 62(3), 033007.

2. Agafonova, N. Y., Aglietta, M., Antonioli, P., Bari, G., Boyarkin, V. V., Bruno,
G., ... & Zichichi, A. (2007). Study of the effect of neutrino oscillations on the
supernova neutrino signal in the LVD detector. Astroparticle Physics, 27(4), 254-270.
"""

def emitted_spectrum(x, flavor, E_tot, hierarchy = 'normal', adiabatic = True):
    F_e = produced_spectrum(x, 'nu_e', E_tot)
    F_ebar = produced_spectrum(x, 'nubar_e', E_tot)
    F_x = produced_spectrum(x, 'nu_x', E_tot)
    if adiabatic:
        U_e3 = np.sqrt(1e-6)
    else:
        U_e3 = np.sqrt(1e-2)
    Ph = np.exp(-U_e3**2 * (delta_m['m_20']/x)**(2/3))
    if hierarchy == 'normal':
        a_e = Ph*U[0,1]**2
        b_e = 0
        c_e = (1 - Ph*U[0,1]**2)
        a_ebar = 0
        b_ebar = U[0,0]**2
        c_ebar = U[0,1]**2
        a_x = 1 - Ph*U[0,0]**2 - Ph*U[0,1]**2 - (1-Ph)*U_e3**2
        b_x = 1 - U[0,0]**2
        c_x = 2 - a_x+1 - b_x+1
    elif hierarchy == 'inverted':
        a_e = U[0,1]**2
        b_e = 0
        c_e = U[0,0]**2
        a_ebar = 0
        b_ebar = Ph*U[0,0]**2
        c_ebar = (1-Ph*U[0,0]**2)
        a_x = 1 - Ph*U[0,1]**2 - (1-Ph)*U_e3**2
        b_x = 1 - U[0,0]**2
        c_x = 2 - a_x+1 - b_x+1
    else:
        raise ValueError('Invalid type of hierarchy, please use "normal" or "inverted"')
    if flavor == 'nu_e':
        return a_e*F_e + b_e*F_ebar + c_e*F_x
    elif flavor == 'nubar_e':
        return a_ebar*F_e + b_ebar*F_ebar + c_ebar*F_x
    elif flavor == 'nu_x':
        return F_x
    else:
        raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')