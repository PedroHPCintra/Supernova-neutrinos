import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Importing csv files from cross section simulations
marley = pd.read_csv('https://raw.githubusercontent.com/PedroHPCintra/Supernova-neutrinos/main/Data/MARLEY_neutrino_argon_CC_cross_section.csv') #MARLEY neutrino-argon
marley_anti = pd.read_csv('https://raw.githubusercontent.com/PedroHPCintra/Supernova-neutrinos/main/Data/MARLEY_anti_neutrino_argon_CC_cross_section.csv') #MARLEY antineutrino-argon
"""
MARLEY: Gardiner, S. (2021). Simulating low-energy neutrino interactions with MARLEY.
Computer Physics Communications, 269, 108123.
"""
oxygen = pd.read_csv('https://raw.githubusercontent.com/PedroHPCintra/Supernova-neutrinos/main/Data/Neutrino_oxigen_cross_section.csv') #neutrino oxygen
"""
Kolbe, E., Langanke, K., & Vogel, P. (2002). Estimates of weak and electromagnetic nuclear
decay signatures for neutrino reactions in Super-Kamiokande. Physical Review D, 66(1), 013007
"""

#################### Cross sections ##########################
def cross_section_nu_e_argon(x):
    """
    Charged current interaction. Electron neutrino and argon nucleus
    """
    f = interp1d(marley['Energy'], marley['Cross-section (1e-38)'],
    fill_value='extrapolate')
    y_new = f(x)
    return y_new*1e-38

def cross_section_nubar_e_argon(x):
    """
    Charged current interaction. Electron antineutrino and argon nucleus
    """
    f = interp1d(marley_anti['Energy'], marley_anti['Cross-section (1e-38)'],
    fill_value='extrapolate')
    y_new = f(x)
    return y_new*1e-38

def cross_section_nu_e_oxygen(x):
    """
    Charged current interaction. Electron neutrino and oxygen nucleus.
    The simulated values are well described by an polynomial function in
    the range from 20 to 100 MeVs (Tomas, R. et al. 2003. Physical Review D).
    """
    if x[0] >= 20 and x[-1] <= 100:
        return 4.7e-40 * (x**0.25 - 15**0.25)**6
    else:
        f = interp1d(oxygen['Energy'], oxygen['16-O(nu-eletron)X (1e-42)'],
                    fill_value='extrapolate', kind='cubic')
        y_new = f(x)
        return y_new*1e-42

def cross_section_nubar_e_oxygen(x):
    """
    Charged current interaction. Electron antineutrino and oxygen nucleus
    """
    f = interp1d(oxygen['Energy'], oxygen['16-O(nubar-positron)X (1e-42)'],
                fill_value='extrapolate', kind = 'cubic')
    y_new = f(x)
    return y_new*1e-42

def cross_section_NC_nu_e(x, threshold, g1, g2):
    """
    Elastic scattering between neutrino of any flavor and an electron
    """
    m_e = 0.511 #MeV
    T_max = (2*(x**2))/(m_e + 2*x)
    sigma_0 = 88.06e-46 #cm^2
    sigma = (sigma_0/m_e)*((g1**2 + g2**2)*(T_max - threshold) - (g2**2 + (g1**2)*(g2**2)*(m_e/(2*x)))*((T_max**2 - threshold**2)/x) + (1/3)*(g2**2)*((T_max**3 - threshold**3)/x**2))
    return sigma

def dsigma_dtheta(theta, E, g1, g2, sigma_0 = 88.06e-46, m_e = 0.511):
    """
    Differential angular cross section of elastic scattering between
    neutrino and electron. The cross section is analytical and depends
    on the scattering angle θ and the neutrino energy E.

    The parameters g1 and g2 specify the neutrino flavor
    """
    Te = (2*m_e*(E**2)*(np.cos(theta)**2))/((m_e + E)**2 - (E**2)*(np.cos(theta)**2))
    top = (E**2)*(E**2 + 2*m_e*E + (m_e**2))*np.cos(theta)*np.sin(theta)
    bottom = ((E**2)*(np.cos(theta)**2)-(E**2)-2*m_e*E-(m_e**2))**2
    part_1 = -top/bottom
    part_2 = g1**2 + g2*(1 - Te/E)**2 - g1*g2*m_e*Te/(E**2)
    f = sigma_0*part_1*part_2
    return f

def cross_section_CC_nu_proton(x, Δ = 1.293):
    """
    Aproximation of IBD cross section valid until 300 MeV.
    Δ is the mass difference between the neutron and the proton Δ = m_n - m_p = 1.293 MeV
    """
    sig_0 = 1e-43 #cm^2
    E_e = x - Δ
    m_e = 0.511
    p_e = np.sqrt(E_e - m_e)
    return sig_0*p_e*E_e*x**(-0.07056 + 0.02018*np.log(x) - 0.001953*(np.log(x)**3))