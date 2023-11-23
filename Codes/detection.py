import numpy as np
import scipy.stats as st
from scipy.integrate import simps
import pandas as pd
from random import choices
from math import gamma
import cross_sections as cs
from constants import *
from emission import *
from scipy.interpolate import interp1d

# Detector efficiency
def efficiency_sigmoid(x, a, b, c):
    """
    100% em 4.5 MeV e 97% em 4.0 MeV
    Fukuda, S., Fukuda, Y., Hayakawa, T., Ichihara, E., Ishitsuka, M., Itow, Y., ... & Ichikawa, Y. (2003). The super-kamiokande detector.
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 501(2-3), 418-462.
    """
    eff = a/(1+np.exp(-b*(x-c)))
    return eff

def resolution_detector(E_det, E_true, detector):
    if detector == 'super-k' or detector == 'Hyper-k':
        delta = 0.5*np.sqrt(E_true)
    elif detector == 'DUNE':
        delta = 0.11*np.sqrt(E_true) + 0.02*E_true
    elif detector == 'JUNO':
        delta = 0.03*np.sqrt(E_true)
    else:
        raise ValueError("Ops, we don't have this detector in our simulations. Try 'super-k', 'Hyper-k', 'DUNE' or 'JUNO'.")
    R = 1/(np.sqrt(2*np.pi)*delta) * np.exp(-1/2*(E_true - E_det)**2 / delta**2)
    return R

def detection_spectra(x, model, flavor = 'nu_e', detector = 'super-k',
                    hierarchy = 'normal', distance = 10, custom_size = False,
                    target_particles = None, get_directions = True,
                    declination = 80.745, right_ascention = -69.75, date = '1987-2-23-23-0-0'):
    if detector == 'super-k':
        E_th = 4.5
        channels = ['ibd','nue_e','nuebar_e','nue_O16','nuebar_O16','numu_e',
                    'numubar_e','nc_nue_O16','nc_nuebar_O16']
        xs_data = cs.snowglobes(channels)
        
        # All these cross sections are in units of 10⁻³⁸ cm²/MeV
        cs_ibd = interp1d(1e3*(10**(xs_data[0][0])), xs_data[0][4]/1e3, fill_value='extrapolate')
        cs_nue_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[1][1]/1e3, fill_value='extrapolate')
        cs_nuebar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[2][4]/1e3, fill_value='extrapolate')
        cs_nue_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[3][1]/1e3, fill_value='extrapolate')
        cs_nuebar_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[4][1]/1e3, fill_value='extrapolate')
        cs_nux_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[5][2]/1e3, fill_value='extrapolate')
        cs_nuxbar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[6][5]/1e3, fill_value='extrapolate')
        cs_nc_nue_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[7][1]/1e3, fill_value='extrapolate')
        cs_nc_nuebar_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[8][4]/1e3, fill_value='extrapolate')
        # Number of target particles
        if custom_size:
            n_target = target_particles
        else:
            n_target = (32e9/18.01528)*6.022e23*2 # number of protons
        # Total cross section
        if get_directions:
            if flavor == 'nu_e':
                ES_cross = cs_nue_e(x)*x*1e-38
                IBD_cross = 0
                O16_cross = cs_nue_O16(x)*x*1e-38# + cs_nc_nue_O16(x)*x*1e-38
            elif flavor == 'nubar_e':
                ES_cross = cs_nuebar_e(x)*x*1e-38
                IBD_cross = cs_ibd(x)*x*1e-38
                O16_cross = cs_nuebar_O16(x)*x*1e-38
            elif flavor == 'nu_x':
                ES_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38
                IBD_cross = 0
                O16_cross = 0
            else:
                raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                    electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                    tau (anti)neutrinos')
        else:
            if flavor == 'nu_e':
                total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_O16(x)*x*1e-38# + cs_nc_nue_O16(x)*x*1e-38
            elif flavor == 'nubar_e':
                total_cross = cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_O16(x)*x*1e-38 + cs_nc_nuebar_O16(x)*x*1e-38
            elif flavor == 'nu_x':
                total_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38
            else:
                raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                    electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                    tau (anti)neutrinos')
        # Detector efficiency
        eff = np.where(x<E_th,0,0.9)
    elif detector == 'DUNE':
        channels = ['ibd','nue_e','nuebar_e','nue_Ar40','nuebar_Ar40','numu_e',
                    'numubar_e','nc_nue_Ar40','nc_nuebar_Ar40','nc_numu_Ar40',
                    'nc_numubar_Ar40']
        xs_data = cs.snowglobes(channels)

        # All these cross sections are in units of 10⁻³⁸ cm²/MeV
        cs_ibd = interp1d(1e3*(10**(xs_data[0][0])), xs_data[0][4]/1e3, fill_value='extrapolate')
        cs_nue_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[1][1]/1e3, fill_value='extrapolate')
        cs_nuebar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[2][4]/1e3, fill_value='extrapolate')
        cs_nue_Ar40 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[3][1]/1e3, fill_value='extrapolate')
        cs_nuebar_Ar40 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[4][1]/1e3, fill_value='extrapolate')
        cs_nux_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[5][2]/1e3, fill_value='extrapolate')
        cs_nuxbar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[6][5]/1e3, fill_value='extrapolate')
        cs_nc_nue_Ar40 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[7][1]/1e3, fill_value='extrapolate')
        cs_nc_nuebar_Ar40 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[8][4]/1e3, fill_value='extrapolate')
        cs_nc_nux_Ar40 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[9][2]/1e3, fill_value='extrapolate')
        cs_nc_nuxbar_Ar40 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[10][5]/1e3, fill_value='extrapolate')
        # Number of target particles
        if custom_size:
            n_target = target_particles
        else:
            n_target = 6.03e32
        # Total cross section
        if flavor == 'nu_e':
            total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_Ar40(x)*x*1e-38 + cs_nc_nue_Ar40(x)*x*1e-38
        elif flavor == 'nubar_e':
            total_cross = cs_nuebar_e(x)*x*1e-38 + cs_nuebar_Ar40(x)*x*1e-38 + cs_nc_nuebar_Ar40(x)*x*1e-38
        elif flavor == 'nu_x':
            total_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38 + cs_nc_nux_Ar40(x)*x*1e-38 + cs_nc_nuxbar_Ar40(x)*x*1e-38
        else:
            raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')
        # Detector efficiency
        eff = efficiency_sigmoid(x, 0.98, 1.2127, 8.0591)
    elif detector == 'JUNO':
        E_th = 0.3
        channels = ['ibd','nue_e','nuebar_e','nue_C12','nuebar_C12','numu_e',
                    'numubar_e','nc_nue_C12','nc_nuebar_C12','nc_numu_C12',
                    'nc_numubar_C12']
        xs_data = cs.snowglobes(channels)

        # All these cross sections are in units of 10⁻³⁸ cm²/MeV
        cs_ibd = interp1d(1e3*(10**(xs_data[0][0])), xs_data[0][4]/1e3, fill_value='extrapolate')
        cs_nue_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[1][1]/1e3, fill_value='extrapolate')
        cs_nuebar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[2][4]/1e3, fill_value='extrapolate')
        cs_nue_C12 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[3][1]/1e3, fill_value='extrapolate')
        cs_nuebar_C12 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[4][1]/1e3, fill_value='extrapolate')
        cs_nux_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[5][2]/1e3, fill_value='extrapolate')
        cs_nuxbar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[6][5]/1e3, fill_value='extrapolate')
        cs_nc_nue_C12 = interp1d(1e3*(10**(xs_data[7][0])), xs_data[7][1]/1e3, fill_value = 'extrapolate')
        cs_nc_nuebar_C12 = interp1d(1e3*(10**(xs_data[8][0])), xs_data[8][4]/1e3, fill_value = 'extrapolate')
        cs_nc_nux_C12 = interp1d(1e3*(10**(xs_data[9][0])), xs_data[9][2]/1e3, fill_value = 'extrapolate')
        cs_nc_nuxbar_C12 = interp1d(1e3*(10**(xs_data[10][0])), xs_data[10][5]/1e3, fill_value = 'extrapolate')
        # Number of target particles
        """"
        Fengpeng, A., Guangpeng, A. N., Qi, A. N., Antonelli, V., Baussan, E.,
        Beacom, J., ... & Sinev, V. (2016). Neutrino physics with JUNO.
        """
        if custom_size:
            n_target = target_particles
        else:
            n_target = 1.5e33
        # Total cross section
        if flavor == 'nu_e':
            total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_C12(x)*x*1e-38 + cs_nc_nue_C12(x)*x*1e-38
        elif flavor == 'nubar_e':
            total_cross = cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_C12(x)*x*1e-38 + cs_nc_nuebar_C12(x)*x*1e-38
        elif flavor == 'nu_x':
            total_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38 + cs_nc_nux_C12(x)*x*1e-38 + cs_nc_nuxbar_C12(x)*x*1e-38
        else:
            raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')
        # Detector efficiency
        eff = np.where(x<E_th,0,1)
    elif detector == 'Hyper-k':
        E_th = 4.5
        channels = ['ibd','nue_e','nuebar_e','nue_O16','nuebar_O16','numu_e',
                    'numubar_e','nc_nue_O16']
        xs_data = cs.snowglobes(channels)

        # All these cross sections are in units of 10⁻³⁸ cm²/MeV
        cs_ibd = interp1d(1e3*(10**(xs_data[0][0])), xs_data[0][4]/1e3, fill_value='extrapolate')
        cs_nue_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[1][1]/1e3, fill_value='extrapolate')
        cs_nuebar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[2][4]/1e3, fill_value='extrapolate')
        cs_nue_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[3][1]/1e3, fill_value='extrapolate')
        cs_nuebar_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[4][1]/1e3, fill_value='extrapolate')
        cs_nux_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[5][2]/1e3, fill_value='extrapolate')
        cs_nuxbar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[6][5]/1e3, fill_value='extrapolate')
        cs_nc_nue_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[7][1]/1e3, fill_value='extrapolate')
        # Number of target particles
        if custom_size:
            n_target = target_particles
        else:
            n_target = (216e9/18.01528)*6.022e23*2 # number of protons
        # Total cross section
        if get_directions:
            if flavor == 'nu_e':
                ES_cross = cs_nue_e(x)*x*1e-38
                IBD_cross = 0
                O16_cross = cs_nue_O16(x)*x*1e-38# + cs_nc_nue_O16(x)*x*1e-38
            elif flavor == 'nubar_e':
                ES_cross = cs_nuebar_e(x)*x*1e-38
                IBD_cross = cs_ibd(x)*x*1e-38
                O16_cross = cs_nuebar_O16(x)*x*1e-38
            elif flavor == 'nu_x':
                ES_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38
                IBD_cross = 0
                O16_cross = 0
            else:
                raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                    electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                    tau (anti)neutrinos')
        else:
            if flavor == 'nu_e':
                total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_O16(x)*x*1e-38# + cs_nc_nue_O16(x)*x*1e-38
            elif flavor == 'nubar_e':
                total_cross = cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_O16(x)*x*1e-38
            elif flavor == 'nu_x':
                total_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38
            else:
                raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                    electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                    tau (anti)neutrinos')
        # Detector efficiency
        eff = np.where(x<E_th,0,0.9) #efficiency_sigmoid(x, 0.918, 1.2127, 3)
    else:
        raise ValueError("Ops, we don't have this detector in our simulations. Try 'super-k', \
            'Hyper-k', 'DUNE' or 'JUNO'.")
    # Normalization
    distance_cm = distance*3.086e21 #conversion from kiloparsec to centimeter
    A = n_target/(4 * np.pi * distance_cm**2)
    # Emission spectrum
    spectrum = earth_spectrum(x, detector, flavor, model, hierarchy, declination, right_ascention, date)
    if get_directions and detector == 'super-k':
        return A*spectrum*O16_cross*eff, A*spectrum*ES_cross*eff, A*spectrum*IBD_cross*eff
    elif get_directions and detector == 'Hyper-k':
        return A*spectrum*O16_cross*eff, A*spectrum*ES_cross*eff, A*spectrum*IBD_cross*eff
    else:
        return A*spectrum*total_cross*eff

# Sampling from the expected detection spectra
def energy_sampler(E, model, detector = 'super-k', hierarchy = 'normal', distance = 10,
                    print_expected = True, only_total_events = False,
                    custom_size = False, target_particles = None, get_directions = True,
                    declination = 80.745, right_ascention = -69.75,
                    date = '1987-2-23-23-0-0', perfect_resolution = False):
    """"
    This function samples individual detected energies for the events based on the
    total number of expected events and the energy spectrum. It applies a random fluctuation
    to the total number of detected events N and another to each individual energy sampled in
    order to simulate detector error.

    E - neutrino energies

    E_tot - Total energy

    detector - detector name

    hierarchy - neutrino mass hierarchy, either "normal" or "inverted"
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
    theta_detec = angle_between(A, B)#*180/np.pi
    # Calculating the number of expected neutrinos from each flavor
    if get_directions:
        spec_O16_ebar, spec_ES_ebar, spec_IBD_ebar = detection_spectra(E, model, 'nubar_e', detector, hierarchy, distance, custom_size, target_particles, get_directions, declination, right_ascention, date)
        spec_O16_e, spec_ES_e, garbage = detection_spectra(E, model, 'nu_e', detector, hierarchy, distance, custom_size, target_particles, get_directions, declination, right_ascention, date)
        spec_ES_x = detection_spectra(E, model, 'nu_x', detector, hierarchy, distance, custom_size, target_particles, get_directions, declination, right_ascention, date)[1]
        N_expected_IBD_ebar = np.round(simps(spec_IBD_ebar, E), 0)
        N_expected_ES_e = np.round(simps(spec_ES_e, E), 0)
        N_expected_ES_ebar = np.round(simps(spec_ES_ebar, E), 0)
        N_expected_ES_x = np.round(simps(spec_ES_x, E), 0)
        N_expected_O16_e = np.round(simps(spec_O16_e, E), 0)
        N_expected_O16_ebar = np.round(simps(spec_O16_ebar, E), 0)

        if print_expected:
            print('\n'
            f'Number of expected neutrinos by flavor at {distance} kpc ({detector})'
            '\n'
            '\n'
            f'electron neutrinos ES: {int(N_expected_ES_e)}'
            '\n'
            f'electron antineutrinos ES: {int(N_expected_ES_ebar)}'
            '\n'
            f'mu/tau (anti)neutrinos ES: {int(N_expected_ES_x)}'
            '\n'
            f'electron neutrinos 16-O: {int(N_expected_O16_e)}'
            '\n'
            f'electron antineutrinos 16-O: {int(N_expected_O16_ebar)}'
            '\n'
            f'electron antineutrinos IBD: {int(N_expected_IBD_ebar)}')
        
        # Sampling                                      
        possible_energies = E
        weights_O16_e = spec_O16_e/N_expected_O16_e
        weights_O16_ebar = spec_O16_ebar/N_expected_O16_ebar
        weights_IBD_ebar = spec_IBD_ebar/N_expected_IBD_ebar
        weights_ES_e = spec_ES_e/N_expected_ES_e
        weights_ES_ebar = spec_ES_ebar/N_expected_ES_ebar
        weights_ES_x = spec_ES_x/N_expected_ES_x

        # Sampling process uses poisson distribution to resample the number of detected neutrinos in each flavor
        N_new_O16_e = np.random.poisson(int(N_expected_O16_e))
        N_new_O16_ebar = np.random.poisson(int(N_expected_O16_ebar))
        N_new_IBD_ebar = np.random.poisson(int(N_expected_IBD_ebar))

        N_new_ES_e = np.random.poisson(int(N_expected_ES_e))
        N_new_ES_ebar = np.random.poisson(int(N_expected_ES_ebar))
        N_new_ES_x = np.random.poisson(int(N_expected_ES_x))
        # If only_total_events = True, the function will only return the total sampled values, not individual energies
        if only_total_events:
            final_samples = {}
            final_samples['16O_nu_e'] = N_new_O16_e
            final_samples['16O_nubar_e'] = N_new_O16_ebar
            final_samples['IBD_nubar_e'] = N_new_IBD_ebar
            final_samples['ES_nu_e'] = N_new_ES_e
            final_samples['ES_nubar_e'] = N_new_ES_ebar
            final_samples['ES_nu_x'] = N_new_ES_x
            total_sample = N_new_IBD_ebar + N_new_O16_e + N_new_O16_ebar + N_new_ES_e + N_new_ES_ebar + N_new_ES_x
            final_samples['Total'] = total_sample
            return final_samples
        else:
            samples_O16_e = choices(possible_energies, weights_O16_e, k = N_new_O16_e)
            samples_O16_ebar = choices(possible_energies, weights_O16_ebar, k = N_new_O16_ebar)
            samples_IBD_ebar = choices(possible_energies, weights_IBD_ebar, k = N_new_IBD_ebar)
            samples_ES_e = choices(possible_energies, weights_ES_e, k = N_new_ES_e)
            samples_ES_ebar = choices(possible_energies, weights_ES_ebar, k = N_new_ES_ebar)
            samples_ES_x = choices(possible_energies, weights_ES_x, k = N_new_ES_x)
            
            ##### Noise in detection energy and neutrino direction #####
            final_samples_ES_e = []
            final_samples_ES_ebar = []
            final_samples_ES_x = []
            final_samples_O16_e = []
            final_samples_O16_ebar = []
            final_samples_IBD_ebar = []

            angles = np.linspace(-np.pi, np.pi, 10000)
            cosines = np.linspace(0, 1, 10000)

            final_thetas_ES_e = []
            final_thetas_ES_ebar = []
            final_thetas_ES_x = []
            final_thetas_O16_e = []
            final_thetas_O16_ebar = []
            final_thetas_IBD_ebar = []

            # Elastic scattering of electron neutrinos
            for i in range(len(samples_ES_e)):
                N = simps(cs.dsigma_dcostheta_ES(cosines, samples_ES_e[i], g1_nu_e, g2_nu_e, sigma_0), cosines)
                weights_angle = cs.dsigma_dcostheta_ES(cosines, samples_ES_e[i], g1_nu_e, g2_nu_e, sigma_0)/N
                rand = np.random.uniform(0, 1)
                if rand >= 0.5:
                    theta = theta_detec + np.arccos(choices(cosines, weights_angle, k = 1)[0])
                elif rand < 0.5:
                    theta = theta_detec - np.arccos(choices(cosines, weights_angle, k = 1)[0])
                if perfect_resolution:
                    new_theta = theta
                else:
                    new_theta = np.random.normal(theta, 26*np.pi/180)
                if new_theta > np.pi:
                    new_theta = new_theta - np.pi
                if new_theta < 0:
                    new_theta = np.pi + new_theta
                final_thetas_ES_e.append(new_theta)

                if perfect_resolution:
                    new_sample_e = samples_ES_e[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_e = np.random.normal(samples_ES_e[i], 0.5*np.sqrt(samples_ES_e[i]))
                    elif detector == 'DUNE':
                        new_sample_e = np.random.normal(samples_ES_e[i], 0.11*np.sqrt(samples_ES_e[i]) + 0.02*samples_ES_e[i])
                    elif detector == 'JUNO':
                        new_sample_e = np.random.normal(samples_ES_e[i], 0.03*np.sqrt(samples_ES_e[i]))
                
                final_samples_ES_e.append(new_sample_e)
            # Elastic scattering of electron antineutrinos
            for i in range(len(samples_ES_ebar)):
                N = simps(cs.dsigma_dcostheta_ES(cosines, samples_ES_ebar[i], g1_barnu_e, g2_barnu_e, sigma_0), cosines)
                weights_angle = cs.dsigma_dcostheta_ES(cosines, samples_ES_ebar[i], g1_barnu_e, g2_barnu_e, sigma_0)/N
                rand = np.random.uniform(0, 1)
                if rand >= 0.5:
                    theta = theta_detec + np.arccos(choices(cosines, weights_angle, k = 1)[0])
                elif rand < 0.5:
                    theta = theta_detec - np.arccos(choices(cosines, weights_angle, k = 1)[0])
                if perfect_resolution:
                    new_theta = theta
                else:
                    new_theta = np.random.normal(theta, 26*np.pi/180)
                if new_theta > np.pi:
                    new_theta = new_theta - np.pi
                if new_theta < 0:
                    new_theta = np.pi + new_theta
                final_thetas_ES_ebar.append(new_theta)

                if perfect_resolution:
                    new_sample_ebar = samples_ES_ebar[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_ebar = np.random.normal(samples_ES_ebar[i], 0.5*np.sqrt(samples_ES_ebar[i]))
                    elif detector == 'DUNE':
                        new_sample_ebar = np.random.normal(samples_ES_ebar[i], 0.11*np.sqrt(samples_ES_ebar[i]) + 0.02*samples_ES_ebar[i])
                    elif detector == 'JUNO':
                        new_sample_ebar = np.random.normal(samples_ES_ebar[i], 0.03*np.sqrt(samples_ES_ebar[i]))

                final_samples_ES_ebar.append(new_sample_ebar)
            # Elastic scattering of Mu/Tau (anti)neutrinos
            for i in range(len(samples_ES_x)):
                N = simps(cs.dsigma_dcostheta_ES(cosines, samples_ES_x[i], g1_nu_x, g2_nu_x, sigma_0), cosines)
                weights_angle = cs.dsigma_dcostheta_ES(cosines, samples_ES_x[i], g1_nu_x, g2_nu_x, sigma_0)/N
                rand = np.random.uniform(0, 1)
                if rand >= 0.5:
                    theta = theta_detec + np.arccos(choices(cosines, weights_angle, k = 1)[0])
                elif rand < 0.5:
                    theta = theta_detec - np.arccos(choices(cosines, weights_angle, k = 1)[0])
                
                if perfect_resolution:
                    new_theta = theta
                else:
                    new_theta = np.random.normal(theta, 26*np.pi/180)
                if new_theta > np.pi:
                    new_theta = new_theta - np.pi
                if new_theta < 0:
                    new_theta = np.pi + new_theta
                final_thetas_ES_x.append(new_theta)

                if perfect_resolution:
                    new_sample_ES_x = samples_ES_x[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_ES_x = np.random.normal(samples_ES_x[i], 0.5*np.sqrt(samples_ES_x[i]))
                    elif detector == 'DUNE':
                        new_sample_ES_x = np.random.normal(samples_ES_x[i], 0.11*np.sqrt(samples_ES_x[i]) + 0.02*samples_ES_x[i])
                    elif detector == 'JUNO':
                        new_sample_ES_x = np.random.normal(samples_ES_x[i], 0.03*np.sqrt(samples_ES_x[i]))

                final_samples_ES_x.append(new_sample_ES_x)
            # Other interactions of electron neutrinos
            for i in range(len(samples_O16_e)):
                v = np.random.uniform(0, 1)
                theta_2 = np.arccos(2*v - 1)
                # theta_2 = np.random.uniform(0, 2*np.pi)
                final_thetas_O16_e.append(theta_2)

                if perfect_resolution:
                    new_sample_e = samples_O16_e[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_e = np.random.normal(samples_O16_e[i], 0.5*np.sqrt(samples_O16_e[i]))
                    elif detector == 'DUNE':
                        new_sample_e = np.random.normal(samples_O16_e[i], 0.11*np.sqrt(samples_O16_e[i]) + 0.02*samples_O16_e[i])
                    elif detector == 'JUNO':
                        new_sample_e = np.random.normal(samples_O16_e[i], 0.03*np.sqrt(samples_O16_e[i]))

                final_samples_O16_e.append(new_sample_e)
            # Other interactions of electron antineutrinos
            for i in range(len(samples_O16_ebar)):
                v = np.random.uniform(0, 1)
                theta_2 = np.arccos(2*v - 1)
                
                # theta_2 = np.random.uniform(0, 2*np.pi)
                final_thetas_O16_ebar.append(theta_2)

                if perfect_resolution:
                    new_sample_ebar = samples_O16_ebar[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_ebar = np.random.normal(samples_O16_ebar[i], 0.5*np.sqrt(samples_O16_ebar[i]))
                    elif detector == 'DUNE':
                        new_sample_ebar = np.random.normal(samples_O16_ebar[i], 0.11*np.sqrt(samples_O16_ebar[i]) + 0.02*samples_O16_ebar[i])
                    elif detector == 'JUNO':
                        new_sample_ebar = np.random.normal(samples_O16_ebar[i], 0.03*np.sqrt(samples_O16_ebar[i]))

                final_samples_O16_ebar.append(new_sample_ebar)
            # Other interactions of Mu/Tau (anti)neutrinos
            for i in range(len(samples_IBD_ebar)):
                # N = simps(cs.P_theta_IBD(angles), angles)
                # weights_angle = cs.P_theta_IBD(angles)/N
                # rand = np.random.uniform(0, 1)
                # if rand >= 0.5:
                #     theta = theta_detec + choices(angles, weights_angle, k = 1)[0]
                #     phi_new = phi_detec + choices(angles, weights_angle, k = 1)[0]
                # else:
                #     theta = theta_detec - choices(angles, weights_angle, k = 1)[0]
                #     phi_new = phi_detec - choices(angles, weights_angle, k = 1)[0]
                # new_theta = np.random.normal(theta, 26*np.pi/180)
                # new_phi = np.random.normal(phi_new, 26*np.pi/180)
                # if new_theta > 2*np.pi:
                #     new_theta = new_theta - 2*np.pi
                # if new_theta < 0:
                #     new_theta = 2*np.pi - new_theta
                # if new_phi > 2*np.pi:
                #     new_phi = new_phi - 2*np.pi
                # if new_phi < 0:
                #     new_phi = 2*np.pi - new_phi
                v = np.random.uniform(0, 1)
                new_theta = np.arccos(2*v - 1)
                # theta_2 = np.random.uniform(0, 2*np.pi)
                final_thetas_IBD_ebar.append(new_theta)

                if perfect_resolution:
                    new_sample_IBD_ebar = samples_IBD_ebar[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_IBD_ebar = np.random.normal(samples_IBD_ebar[i], 0.5*np.sqrt(samples_IBD_ebar[i]))
                    elif detector == 'DUNE':
                        new_sample_IBD_ebar = np.random.normal(samples_IBD_ebar[i], 0.11*np.sqrt(samples_IBD_ebar[i]) + 0.02*samples_IBD_ebar[i])
                    elif detector == 'JUNO':
                        new_sample_IBD_ebar = np.random.normal(samples_IBD_ebar[i], 0.03*np.sqrt(samples_IBD_ebar[i]))

                final_samples_IBD_ebar.append(new_sample_IBD_ebar)
            
            final_samples_ES_e = np.array(final_samples_ES_e)
            final_samples_ES_ebar = np.array(final_samples_ES_ebar)
            final_samples_ES_x = np.array(final_samples_ES_x)
            final_samples_O16_e = np.array(final_samples_O16_e)
            final_samples_O16_ebar = np.array(final_samples_O16_ebar)
            final_samples_IBD_ebar = np.array(final_samples_IBD_ebar)

            final_thetas_ES_e = np.array(final_thetas_ES_e)
            final_thetas_ES_ebar = np.array(final_thetas_ES_ebar)
            final_thetas_ES_x = np.array(final_thetas_ES_x)
            final_thetas_O16_e = np.array(final_thetas_O16_e)
            final_thetas_O16_ebar = np.array(final_thetas_O16_ebar)
            final_thetas_IBD_ebar = np.array(final_thetas_IBD_ebar)

            # Organizing outputs
            final_samples = {}
            final_samples['Energy_ES_nu_e'] = final_samples_ES_e
            final_samples['Energy_ES_nubar_e'] = final_samples_ES_ebar
            final_samples['Energy_ES_nu_x'] = final_samples_ES_x
            total_sample_ES = np.concatenate((final_samples_ES_e,final_samples_ES_ebar,
                                        final_samples_ES_x))
            final_samples['Energy_ES_Total'] = total_sample_ES
            final_samples['Energy_16O_nu_e'] = final_samples_O16_e
            final_samples['Energy_16O_nubar_e'] = final_samples_O16_ebar
            final_samples['Energy_IBD_nubar_e'] = final_samples_IBD_ebar
            total_sample_non_ES = np.concatenate((final_samples_O16_e,final_samples_O16_ebar,
                                        final_samples_IBD_ebar))
            final_samples['Energy_Non-ES_Total'] = total_sample_non_ES
            total_sample = np.concatenate((total_sample_ES, total_sample_non_ES))
            final_samples['Energy_Total'] = total_sample

            final_samples['Angle_theta_ES_nu_e'] = final_thetas_ES_e
            final_samples['Angle_theta_ES_nubar_e'] = final_thetas_ES_ebar
            final_samples['Angle_theta_ES_nu_x'] = final_thetas_ES_x
            total_thetas_ES = np.concatenate((final_thetas_ES_e,final_thetas_ES_ebar,
                                        final_thetas_ES_x))
            final_samples['Angle_theta_ES_Total'] = total_thetas_ES
            final_samples['Angle_theta_16O_nu_e'] = final_thetas_O16_e
            final_samples['Angle_theta_16O_nubar_e'] = final_thetas_O16_ebar
            final_samples['Angle_theta_IBD_nubar_e'] = final_thetas_IBD_ebar
            total_thetas_non_ES = np.concatenate((final_thetas_O16_e,final_thetas_O16_ebar,
                                        final_thetas_IBD_ebar))
            final_samples['Angle_theta_Non-ES_Total'] = total_thetas_non_ES
            total_thetas = np.concatenate((total_thetas_ES, total_thetas_non_ES))
            final_samples['Angle_theta_Total'] = total_thetas
            final_samples['Nadir'] = np.pi/2 - alpha
            final_samples['Zenith'] = theta_detec
            return final_samples
    else:
        spec_nue = detection_spectra(E, model, 'nu_e', detector, hierarchy, distance, custom_size, target_particles, get_directions, declination, right_ascention, date)
        spec_nubare = detection_spectra(E, model, 'nubar_e', detector, hierarchy, distance, custom_size, target_particles, get_directions, declination, right_ascention, date)
        spec_nux = detection_spectra(E, model, 'nu_x', detector, hierarchy, distance, custom_size, target_particles, get_directions, declination, right_ascention, date)
        N_expected_e = np.round(simps(spec_nue, E), 0)
        N_expected_ebar = np.round(simps(spec_nubare, E), 0)
        N_expected_x = np.round(simps(spec_nux, E), 0)

        if print_expected:
            print('\n'
            f'Number of expected neutrinos by flavor at {distance} kpc ({detector})'
            '\n'
            '\n'
            f'electron neutrinos: {int(N_expected_e)}'
            '\n'
            f'electron antineutrinos: {int(N_expected_ebar)}'
            '\n'
            f'mu/tau (anti)neutrinos: {int(N_expected_x)}')
        
        # Sampling
        possible_energies = E
        weights_e = spec_nue/N_expected_e
        weights_ebar = spec_nubare/N_expected_ebar
        weights_x = spec_nux/N_expected_x

        # Sampling process uses poisson distribution to resample the number of detected neutrinos in each flavor
        N_new_e = np.random.poisson(int(N_expected_e))
        N_new_ebar = np.random.poisson(int(N_expected_ebar))
        N_new_x = np.random.poisson(int(N_expected_x))

        # If only_total_events = True, the function will only return the total sampled values, not individual energies
        if only_total_events:
            final_samples = {}
            final_samples['nu_e'] = N_new_e
            final_samples['nubar_e'] = N_new_ebar
            final_samples['nu_x'] = N_new_x
            total_sample = N_new_x + N_new_e + N_new_ebar
            final_samples['Total'] = total_sample
            return final_samples
        else:
            samples_e = choices(possible_energies, weights_e, k = N_new_e)
            samples_ebar = choices(possible_energies, weights_ebar, k = N_new_ebar)
            samples_x = choices(possible_energies, weights_x, k = N_new_x)
            
            ##### Noise in detection energy and neutrino direction #####
            final_samples_e = []
            final_samples_ebar = []
            final_samples_x = []

            # Electron neutrinos
            for i in range(len(samples_e)):
                if perfect_resolution:
                        new_sample_e = samples_e[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_e = np.random.normal(samples_e[i], 0.5*np.sqrt(samples_e[i]))
                    elif detector == 'DUNE':
                        new_sample_e = np.random.normal(samples_e[i], 0.11*np.sqrt(samples_e[i]) + 0.02*samples_e[i])
                    elif detector == 'JUNO':
                        new_sample_e = np.random.normal(samples_e[i], 0.03*np.sqrt(samples_e[i]))

                final_samples_e.append(new_sample_e)
            # Electron antineutrinos
            for i in range(len(samples_ebar)):
                if perfect_resolution:
                    new_sample_ebar = samples_ebar[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_ebar = np.random.normal(samples_ebar[i], 0.5*np.sqrt(samples_ebar[i]))
                    elif detector == 'DUNE':
                        new_sample_ebar = np.random.normal(samples_ebar[i], 0.11*np.sqrt(samples_ebar[i]) + 0.02*samples_ebar[i])
                    elif detector == 'JUNO':
                        new_sample_ebar = np.random.normal(samples_ebar[i], 0.03*np.sqrt(samples_ebar[i]))

                final_samples_ebar.append(new_sample_ebar)
            # Mu/Tau (anti)neutrinos
            for i in range(len(samples_x)):
                if perfect_resolution:
                    new_sample_x = samples_x[i]
                else:
                    if detector == 'super-k' or detector == 'Hyper-k':
                        new_sample_x = np.random.normal(samples_x[i], 0.5*np.sqrt(samples_x[i]))
                    elif detector == 'DUNE':
                        new_sample_x = np.random.normal(samples_x[i], 0.11*np.sqrt(samples_x[i]) + samples_x[i])
                    elif detector == 'JUNO':
                        new_sample_x = np.random.normal(samples_x[i], 0.03*np.sqrt(samples_x[i]))

                final_samples_x.append(new_sample_x)
            
            final_samples_e = np.array(final_samples_e)
            final_samples_ebar = np.array(final_samples_ebar)
            final_samples_x = np.array(final_samples_x)

            # Organizing outputs
            final_samples = {}
            final_samples['Energy_nu_e'] = final_samples_e
            final_samples['Energy_nubar_e'] = final_samples_ebar
            final_samples['Energy_nu_x'] = final_samples_x
            total_sample = np.concatenate((final_samples_e,final_samples_ebar,final_samples_x))
            final_samples['Energy_Total'] = total_sample
            # final_samples['Energy_nu_e'] = samples_e
            # final_samples['Energy_nubar_e'] = samples_ebar
            # final_samples['Energy_nu_x'] = samples_x
            # total_sample = np.concatenate((samples_e,samples_ebar,samples_x))
            # final_samples['Energy_Total'] = total_sample
            return final_samples