import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simps
from constants import V_ckm, G_F

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

# SnowGlobes cross sections
def snowglobes(channels):
    xs_data = []
    columns=['Log(E_cs [Mev])','nu_e_cs[10^-38 cm^2/GeV]', 'nu_mu_cs','nu_tau_cs','nu_e_bar_cs',
            'nu_mu_bar_cs','nu_tau_bar_cs']
    for i in range(len(channels)):
        folder = f'Data/xscns/xs_{channels[i]}.dat'
        df = pd.read_csv(folder,delim_whitespace=True,skiprows=[0,1], names=columns)
        xs_data.append([])
        for col in columns:
            xs_data[i].append(np.asarray(df[col]))
    return xs_data

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
        for i in range(len(y_new)):
            if y_new[i] < 0:
                y_new[i] = 1.89e-8
        return y_new*1e-42

def cross_section_nubar_e_oxygen(x):
    """
    Charged current interaction. Electron antineutrino and oxygen nucleus
    """
    f = interp1d(oxygen['Energy'], oxygen['16-O(nubar-positron)X (1e-42)'],
                fill_value='extrapolate', kind = 'cubic')
    y_new = f(x)
    for i in range(len(y_new)):
        if y_new[i] < 0:
            y_new[i] = 1.89e-8
    return y_new*1e-42

def cross_section_NC_nu_e(x, threshold, g1, g2):
    """
    Elastic scattering between neutrino of any flavor and an electron
    """
    m_e = 0.511 #MeV
    T_max = (2*(x**2))/(m_e + 2*x)
    sigma_0 = 88.06e-46 #cm^2
    sigma = (sigma_0/m_e)*((g1**2 + g2**2)*(T_max - threshold) - (g2**2 + (g1**2)*(g2**2)*(m_e/(2*x)))*((T_max**2 - threshold**2)/x) + (1/3)*(g2**2)*((T_max**3 - threshold**3)/x**2))
    for i in range(len(sigma)):
        if sigma[i] < 0:
            sigma[i] = 1e-51
    return sigma

def dsigma_dtheta_ES(theta, E, g1, g2, sigma_0 = 88.06e-46, m_e = 0.511):
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

def dsigma_dcostheta_ES(cos_theta, E, g1, g2, sigma_0 = 88.06e-46, m_e = 0.511):
    Te = (2*m_e*(E**2)*(cos_theta**2))/((m_e + E)**2 - (E**2)*(cos_theta**2))
    part_1 = (4*(E**2)*(m_e + E)**2 * cos_theta)/(((m_e + E)**2 - (E**2)*(cos_theta**2))**2)
    part_2 = g1**2 + (g2**2)*(1 - Te/E)**2 - g1*g2*m_e*Te/(E**2)
    f = sigma_0*part_1*part_2/m_e
    return f

def dsigma_dsintheta_ES(sin_theta, E, g1, g2, sigma_0 = 88.06e-46, m_e = 0.511):
    Te = (2*m_e*(E**2)*(1 - sin_theta**2))/((m_e + E)**2 - (E**2)*(1 - sin_theta)**2)
    part1 = -(4*m_e*(E**2)*(E**2 + 2*m_e*E + m_e**2)*sin_theta)/(((E**2)*sin_theta**2 + 2*m_e*E + m_e**2)**2)
    part2 = g1**2 + (g2**2)*(1 - Te/E)**2 - g1*g2*m_e*Te/(E**2)
    f = sigma_0*part1*part2/m_e
    return f

def P_theta_IBD(theta):
    return 1 - 0.102*np.cos(theta)

def cross_section_ES_nu_proton(x, flavor):
    if 'nubar' in flavor:
        ca = -1.27/2
    else:
        ca = 1.27/2
    m_p = 938.27
    cv = 0.04
    T_max = 2*x**2 / (m_p + 2*x)
    T = np.linspace(0, T_max, 10000)
    dsigma_dT = (m_p * G_F**2 / (2*np.pi* x**2))*((cv + ca)**2 * x**2 + (cv - ca)**2 * (x - T)**2 - (cv**2 - ca**2)*m_p*T)
    sigma = simps(dsigma_dT, T)
    return sigma

def cross_section_CC_nu_proton(x, Δ = 1.293):
    """
    Aproximation of IBD cross section valid until 300 MeV.
    Δ is the mass difference between the neutron and the proton Δ = m_n - m_p = 1.293 MeV

    Ref:
        Strumia, A., & Vissani, F. (2003). Precise quasielastic neutrino/nucleon
        cross-section. Physics Letters B, 564(1-2), 42-54.
    """
    sig_0 = 1e-43 #cm^2
    E_e = x - Δ
    m_e = 0.511
    p_e = np.sqrt(E_e - m_e)
    return sig_0*p_e*E_e*x**(-0.07056 + 0.02018*np.log(x) - 0.001953*(np.log(x)**3))

def F2helm(q2, A):
   """
   returns helm form factor.
   input:
      q2 - squared momentum transfer in keV^2
      A - atomic mass number
   """
   hbarc=1.97e5 # keV fm
   s=0.9/hbarc
   a=0.52/hbarc
   c=(1.23*A**(1./3.)-0.6)/hbarc
   rn=np.sqrt(c**2+7./3.*np.pi**2*a**2-5*s**2)
   q=np.sqrt(q2)
   F2 = (3*(np.sin(q*rn)-q*rn*np.cos(q*rn))/(q*rn)**3*np.exp(-q2*s**2/2))**2
   F2 = np.clip(F2, 0, 1e30)
   return F2

def diff_cross_section_CEnuNS(ER, Enu, mN, AN, ZN):
    """
    returns the differential scattering cross section
    per nuclear recoil energy ER for a neutrino with
    energy Enu scattering elastically (CEvNS) off a
    nucleus with N
    input:
       ER - nuclear recoil energy in [keV]
       Enu - neutrino energy in [MeV]
       mN - mass of N in [GeV]
       AN - atomic number of N
       ZN - number of protons in N
    output:
       differential cross section in [cm^2/keV]
    """
    # constants
    GF = 1.1663787e-5  # Fermi constant in [1/GeV^2]
    sin2W = 0.23121  # weak mixing angle
    hbarc_GeVcm = 1.97e-14  # in [GeV cm]
    # unit conversion
    GeVPERkeV = 1e-6
    #
    unitconv = hbarc_GeVcm ** 2 * GeVPERkeV
    prefac = GF ** 2 / (4.0 * np.pi)
    QN2 = (AN - ZN - (1.0 - 4.0 * sin2W) * ZN) ** 2
    xsec = (
        QN2 * mN * (1.0 - mN * ER / (2.0 * Enu ** 2)) * F2helm(2.0 * 1e6 * mN * ER, AN)
    )
    xsec = np.clip(xsec, 0, 1e30)
    return unitconv * prefac * xsec

def cross_section_CEnuNS(Enu, mN, AN, ZN):
    cs = []
    for i in range(len(Enu)):
        ER = np.linspace(0, 2*Enu[i]/(mN*1e3), len(Enu))
        piece = simps(diff_cross_section_CEnuNS(ER, Enu[i], mN, AN, ZN), ER)
        print(piece)
        cs.append(piece)
    cs = np.array(cs)
    return cs

# E = np.linspace(0, 100, 10)
# c = cross_section_CEnuNS(E, 25, 10, 10)
# print(c)