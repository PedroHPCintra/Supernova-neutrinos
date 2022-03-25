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
Dighe, A. S., & Smirnov, A. Y. (2000). Identifying the neutrino mass
spectrum from a supernova neutrino burst. Physical Review D, 62(3), 033007.
"""

def emitted_spectrum(x, flavor, E_tot, hierarchy = 'normal'):
    F_e = produced_spectrum(x, 'nu_e', E_tot)
    F_ebar = produced_spectrum(x, 'nubar_e', E_tot)
    F_x = produced_spectrum(x, 'nu_x', E_tot)
    if hierarchy == 'normal':
        Ph = 1
        Pl = 0
        Pl_bar = 0
    elif hierarchy == 'inverted':
        Ph = 0
        Pl = 1
        Pl_bar = 0
    else:
        raise ValueError('Invalid type of hierarchy, please use "normal" or "inverted"')
    p = np.abs(U[0,0])**2 * Ph*Pl + np.abs(U[0,1])**2 * (Ph - Ph*Pl) + np.abs(U[0,2])**2 * (1-Ph)
    pbar = np.abs(U[0,0])**2 * (1-Pl_bar) + np.abs(U[0,1])**2 * Pl_bar
    if flavor == 'nu_e':
        return p*F_e + (1-p)*F_x
    elif flavor == 'nubar_e':
        return pbar*F_ebar + (1-pbar)*F_x
    elif flavor == 'nu_x':
        return (1/4)*((1-p)*F_e + (1-pbar)*F_ebar + (2 + p + pbar)*F_x)
    else:
        raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')