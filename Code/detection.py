import numpy as np
import scipy.stats as st
from scipy.integrate import simps
import pandas as pd
from random import choices
from math import gamma
import cross_sections as cs
from constants import *
from emission import spectra

def efficiency_super_k(x, a, b, c):
    """
    100% em 4.5 MeV e 97% em 4.0 MeV
    Fukuda, S., Fukuda, Y., Hayakawa, T., Ichihara, E., Ishitsuka, M., Itow, Y., ... & Ichikawa, Y. (2003). The super-kamiokande detector.
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 501(2-3), 418-462.
    """
    eff = a/(1+np.exp(-b*(x-c)))
    return eff

E = np.linspace(0.01, 100, 1000000)
n_water = (32000000000/18.01528)*6.022e23*2

a = spectra(E, 4, 1e53)

def detection_spectra(x, alpha, E_mean, E_tot):
    distance = 10000
    n_target =  n_water
    distance_cm = distance*3.09e16
    A = n_target/(4 * np.pi * distance_cm**2)
    cross_oxygen = cs.cross_section_nu_e_oxygen(x)
    cross_oxygen_anti = cs.cross_section_nubar_e_oxygen(x)
    cross_scatter = cs.cross_section_NC_nu_e(x, 5, g1_nu_e, g2_nu_e)
    cross_scatter_anti = cs.cross_section_NC_nu_e(x, 5, g1_barnu_e, g2_barnu_e)
    cross_scatter_x = cs.cross_section_NC_nu_e(x, 5, g1_nu_x, g2_nu_x)
    cross_scatter_anti_x = cs.cross_section_NC_nu_e(x, 5, g1_barnu_x, g2_barnu_x)
    cross_ivb = cs.cross_section_CC_nu_proton(x)
    total_cross = cross_oxygen + cross_oxygen_anti + cross_scatter + cross_scatter_anti + cross_scatter_x + cross_scatter_anti_x + cross_ivb
    spectrum = spectra(x, alpha, E_mean, E_tot)
    return A*spectrum*total_cross*efficiency_super_k(x, 100, 7, 3.5)