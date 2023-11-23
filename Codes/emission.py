import numpy as np
import scipy.stats as st
from scipy.integrate import simps
from scipy.interpolate import interp1d
import datetime
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
import pandas as pd
from random import choices
from math import gamma
from constants import *
import scipy.linalg as la

#Spectrum of neutrinos produced in the star
def produced_spectrum(x, specie, model):
    try:
        if specie == 'nu_e':
            alpha = params[model][0]
            E_mean = params[model][3]
            E_tot = params[model][6]
        elif specie == 'nubar_e':
            alpha = params[model][1]
            E_mean = params[model][4]
            E_tot = params[model][7]
        elif specie == 'nu_x':
            alpha = params[model][2]
            E_mean = params[model][5]
            E_tot = params[model][8]
        else:
            raise KeyError('Invalid entry for neutrino species. Please use "nu_e" for '
                'electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or '
                'tau (anti)neutrinos')
    except KeyError:
        raise KeyError('Invalid supernova model. Try "LS220-27.0", "LS220-11.2", "Shen-27.0" or "Shen-11.2"')
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
data_ea = np.loadtxt('Data/Earth_density_profile_kg_per_m3.txt', delimiter=';', skiprows=4)
r_data_earth = data_ea[:,0] #km
rho_earth = data_ea[:,1]*1e3/1e6

f_density = interp1d(r_data_earth, rho_earth, fill_value = 'extrapolate')

def nu_path(nadir):
    L = R_earth*np.sqrt(2*(1-np.cos(2*nadir)))
    path = np.linspace(0, L, 10000)
    rho_path = []
    for i in range(len(path)):
        r = np.sqrt(R_earth**2 + path[i]**2 - 2*R_earth*path[i]*np.cos(np.pi/2 - nadir))
        rho_path.append(f_density(r))
    rho_path = np.array(rho_path)
    return path, rho_path

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def earth_path(detector: str, date: str, declination, right_ascention):
    """
    Date format (year-month-day-hour-minute-second)
    """
    z = R_earth*np.cos(np.pi/2 - detectors_lat[detector])
    x = R_earth*np.sin(np.pi/2 - detectors_lat[detector])*np.cos(detectors_lon[detector])
    y = R_earth*np.sin(np.pi/2 - detectors_lat[detector])*np.sin(detectors_lon[detector])

    detec_time = datetime.datetime(int(date.split('-')[0]),int(date.split('-')[1]),int(date.split('-')[2]),
                                   int(date.split('-')[3]),int(date.split('-')[4]),int(date.split('-')[5]))
    observing_location = EarthLocation(lat=0*u.deg, lon=0*u.deg)
    observing_time = Time(detec_time, scale='utc', location=observing_location)
    GMST = observing_time.sidereal_time('mean')
    GMST_deg = GMST.hour*15
    nu_lat = declination*np.pi/180
    nu_lon = (right_ascention - GMST_deg)*np.pi/180
    z2 = R_earth*np.cos(np.pi/2 - nu_lat)
    x2 = R_earth*np.sin(np.pi/2 - nu_lat)*np.cos(nu_lon)
    y2 = R_earth*np.sin(np.pi/2 - nu_lat)*np.sin(nu_lon)

    A = [x, y, z]
    B = [x2, y2, z2]

    alpha = angle_between(A, np.subtract(A, B))#*180/np.pi
    nadir = np.pi/2 - alpha
    path_length, path_dens = nu_path(nadir)
    return path_length, path_dens

def ne_func_earth(r, ne_f):
    if type(r) == np.ndarray:
        ne_array = np.where(ne_f(r)<0, 0, ne_f(r))
    else:
        if ne_f(r) < 0:
            ne_array = 0
        else:
            ne_array = ne_f(r)
    return ne_array

def emitted_spectrum(x, flavor, model, hierarchy = 'normal'):
    F_e = produced_spectrum(x, 'nu_e', model)
    F_ebar = produced_spectrum(x, 'nubar_e', model)
    F_x = produced_spectrum(x, 'nu_x', model)
    Y_e = 0.5
    A = 1e10
    M_N = 980
    E_na = (np.pi/12)**(3/2) * (delta_m['m_21']*s_12**3/c_12**2)*np.sqrt(2*np.sqrt(2)*G_F*1e-11*Y_e*A/M_N)
    P_H = np.exp(-(E_na/x)**(2/3))
    if hierarchy == 'normal':
        a_e = P_H*np.abs(U[0,1])**2
        b_e = 0
        c_e = (1 - P_H*np.abs(U[0,1])**2)
        a_ebar = np.abs(U[0,0])**2
        b_ebar = 0
        c_ebar = np.abs(U[0,1])**2
        a_x = 0.25*(1 - np.abs(U[0,1])**2)*P_H
        b_x = 0.25*np.abs(U[0,1])**2
        c_x = 0.25*(3 - np.abs(U[0,1])**2  + P_H*np.abs(U[0,1])**2)
    elif hierarchy == 'inverted':
        a_e = np.abs(U[0,1])**2
        b_e = 0
        c_e = np.abs(U[0,0])**2
        a_ebar = P_H*np.abs(U[0,0])**2
        b_ebar = 0
        c_ebar = (1-P_H*np.abs(U[0,1])**2)
        a_x = 0.25*(1 - np.abs(U[0,0])**2)
        b_x = 0.25*P_H*np.abs(U[0,1])**2
        c_x = 0.25*(1 - a_x + 1 - b_x)
    else:
        raise KeyError('Invalid type of hierarchy, please use "normal" or "inverted"')
    if flavor == 'nu_e':
        return a_e*F_e + b_e*F_ebar + c_e*F_x
    elif flavor == 'nubar_e':
        return a_ebar*F_e + b_ebar*F_ebar + c_ebar*F_x
    elif flavor == 'nu_x':
        return a_x*F_e + b_x*F_ebar + c_x*F_x
    else:
        raise KeyError('Invalid entry for neutrino species. Please use "nu_e" for '
                'electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or '
                'tau (anti)neutrinos')
    
def Pje_Earth(E, nu_path, path_dens, mix, j, nu_nubar):
    E = E*1e6 #eV
    n = 801
    Pee_list = []
    #imaginary number
    i = 1j
    k = 0
    ne_f = interp1d(nu_path, path_dens, kind='slinear', fill_value='extrapolate')

    #Energy in MeV and rho in g cm^-3
    def diag(E, Vcc0):
        #Re-evaluating the M2 with the E parameter for the function
        if mix == 'normal':
            M2 = np.array([[0,0,0],[0,delta_m['m_10'],0],[0,0,delta_m['m_20']]])
        elif mix == 'inverted':
            M2 = np.array([[0,0,0],[0,delta_m['m_10'],0],[0,0,-delta_m['m_20']]])
        if nu_nubar == 'nuebar':
            Vcc0 = -Vcc0

        #Hamiltonian in flavour basis
        M2f = U3 @ M2 @ U3_dag
        
        #Potential in matter
        Vcc = np.array([[Vcc0,0.,0.],[0.,0.,0.],[0.,0.,0.]])

        Hf = 1/(2*E) * M2f + Vcc

        #eigenvalues
        eigvals, eigvecs = la.eig(Hf)
        eigvals = eigvals.real

        #sorting eigenvalues list
        id_sor = np.argsort(abs(eigvals))

        #adding eigenvalues to a list
        eval1 = eigvals[id_sor[0]]
        eval2 = eigvals[id_sor[1]]
        eval3 = eigvals[id_sor[2]]

        #collecting eigenvectors from sorted eigenvalues
        eve1 = eigvecs[:,id_sor[0]]
        eve2 = eigvecs[:,id_sor[1]]
        eve3 = eigvecs[:,id_sor[2]]

        #Eigenvector for electron neutrino spectrum
        Ue1 = (eve1[0])
        Ue2 = (eve2[0])
        Ue3 = (eve3[0])
        #Eigenvector for muon neutrino spectrum
        Umu1 = (eve1[1])
        Umu2 = (eve2[1])
        Umu3 = (eve3[1])
        #Eigenvector for tau neutrino spectrum
        Utau1 = (eve1[2])
        Utau2 = (eve2[2])
        Utau3 = (eve3[2])

        Um3 = np.array([[Ue1,Ue2,Ue3],
                     [Umu1,Umu2,Umu3],
                     [Utau1,Utau2,Utau3]])
        Um3_dag = np.transpose(np.conjugate(Um3))

        return eval1, eval2, eval3, Um3, Um3_dag

    #initial radius of the Earth
    r0 = min(nu_path) #km
    #final radius to calculate Pee
    rf = max(nu_path) #km
    r_list = np.linspace(r0, rf, n) #km
    Pee_list = []
    k = 0

    #loop over all points in the SN potential
    for r in r_list[1:]:
        vcc0 = np.sqrt(2) * Gf * ne_func_earth(r, ne_f)
        id = list(r_list).index(r)
        Delta_x_km = r - r_list[id-1]
        Delta_x_eV = Delta_x_km/(0.197e9 * 1e-15 / 1000) #eV^-1

        eval1, eval2, eval3, Um, Um_dag = diag(E, vcc0)
        exp_Hm = np.array([[np.exp(-i*eval1*Delta_x_eV), 0, 0],
                        [0, np.exp(-i*eval2*Delta_x_eV), 0],
                        [0, 0, np.exp(-i*eval3*Delta_x_eV)]])
        if k == 0:
            U = Um @ exp_Hm @ Um_dag
            nue_0 = np.array([1,0,0])
            numu_0 = np.array([0,1,0])
            nutau_0 = np.array([0,0,1])
            if j == '1':
                nu1_0 = Um_dag[0,0] * nue_0 + Um_dag[0,1] * numu_0 + Um_dag[0,2] * nutau_0
                nui = U @ nu1_0
            elif j == '2':
                nu2_0 = Um_dag[1,0] * nue_0 + Um_dag[1,1] * numu_0 + Um_dag[1,2] * nutau_0
                nui = U @ nu2_0
            elif j == '3':
                nu3_0 = Um_dag[2,0] * nue_0 + Um_dag[2,1] * numu_0 + Um_dag[2,2] * nutau_0
                nui = U @ nu3_0
            
        else:
            U = Um @ exp_Hm @ Um_dag
            nui = U @ nui

        nue_0 = np.array([1,0,0])
        Pee = abs(nui @ nue_0)**2
        Pee_list.append(Pee)
        k = k+1

    return Pee
    
def earth_spectrum(x, detector, flavor, model, hierarchy = 'normal', declination = 80.745, right_ascention = -69.75, date:str = '1987-2-23-23-0-0'):
    emit_nu_e = emitted_spectrum(x, 'nu_e', model, hierarchy)
    emit_nu_x = emitted_spectrum(x, 'nu_x', model, hierarchy)
    emit_nubar_e = emitted_spectrum(x, 'nubar_e', model, hierarchy)
    path_length, path_dens = earth_path(detector, date, declination, right_ascention)
    P_1e = []
    P_2e = []
    P_3e = []
    P_1ebar = []
    P_2ebar = []
    P_3ebar = []
    if hierarchy == 'normal':
        for i in range(len(x)):
            E_new = x[i]
            P_3e.append(Pje_Earth(E=E_new, nu_path = path_length, path_dens = path_dens, mix=hierarchy, j='3', nu_nubar='nue'))
            P_1ebar.append(Pje_Earth(E=E_new, nu_path = path_length, path_dens = path_dens, mix=hierarchy, j='1', nu_nubar='nuebar'))
    elif hierarchy == 'inverted':
        for i in range(len(x)):
            E_new = x[i]
            P_2e.append(Pje_Earth(E=E_new, nu_path = path_length, path_dens = path_dens, mix=hierarchy, j='2', nu_nubar='nue'))
            P_3ebar.append(Pje_Earth(E=E_new, nu_path = path_length, path_dens = path_dens, mix=hierarchy, j='3', nu_nubar='nuebar'))
    else:
        raise ValueError('The only allowed inputs for hierarchy are either "normal" or "inverted".')
    
    P_1e = np.array(P_1e)
    P_2e = np.array(P_2e)
    P_3e = np.array(P_3e)
    P_1ebar = np.array(P_1ebar)
    P_2ebar = np.array(P_2ebar)
    P_3ebar = np.array(P_3ebar)
    if hierarchy == 'normal':
        if flavor == 'nu_e':
            return emit_nu_e*P_3e + emit_nu_x*(1 - P_3e)
        elif flavor == 'nu_x':
            return emit_nu_e*(1 - P_3e)/2 + emit_nu_x*(2 + P_3e + P_1ebar)/2 + emit_nubar_e*(1 - P_1ebar)/2
        elif flavor == 'nubar_e':
            return emit_nubar_e*P_1ebar + emit_nu_x*(1 - P_1ebar)
    elif hierarchy == 'inverted':
        if flavor == 'nu_e':
            return emit_nu_e*P_2e + emit_nu_x*(1 - P_2e)
        elif flavor == 'nu_x':
            return emit_nu_e*(1 - P_2e)/2 + emit_nu_x*(2 + P_2e + P_3ebar)/2 + emit_nubar_e*(1 - P_3ebar)/2
        elif flavor == 'nubar_e':
            return emit_nubar_e*P_3ebar + emit_nu_x*(1 - P_3ebar)

