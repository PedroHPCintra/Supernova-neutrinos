import numpy as np
import scipy.stats as st
from scipy.integrate import simps
import pandas as pd
from random import choices
from numba import njit
from math import gamma
from cross_sections import *
from constants import *

def spectra(x, α, E_mean, E_tot):
    A = E_tot*((1+α)**(1+α))/((gamma(1+α))*E_mean**(2+α))
    f_nu = A*(x**α)*np.exp(-(α + 1)*x/E_mean)
    return f_nu

def efficiency_super_k(x, a, b, c):
    """
    100% em 4.5 MeV e 97% em 4.0 MeV
    Fukuda, S., Fukuda, Y., Hayakawa, T., Ichihara, E., Ishitsuka, M., Itow, Y., ... & Ichikawa, Y. (2003). The super-kamiokande detector.
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 501(2-3), 418-462.
    """
    eff = a/(1+np.exp(-b*(x-c)))
    return eff

E = np.linspace(0.01, 100, 1000000)
n_water = (22500000000/18.01528)*6.022e23*2

#@njit
def detection_spectra(x, α, E_mean, E_tot):
    distance = 10000
    n_target =  (22500000000/18.01528)*6.022e23*2
    distance_cm = distance*3.09e16
    A = n_target/(4 * np.pi * distance_cm**2)
    return A*spectra(x, α, E_mean, E_tot)*(cross_section_CC_nu_proton(x) + cross_section_NC_nu_e(x, 0, g1_barnu_e, g2_barnu_e))*efficiency(x, 100, 7, 3.5)