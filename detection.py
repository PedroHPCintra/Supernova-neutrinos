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

def detection_spectra(x, E_tot, flavor = 'nu_e', detector = 'super-k',
                    hierarchy = 'normal', distance = 10, phi = 0):
    if detector == 'super-k':
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
        n_target = (32e9/18.01528)*6.022e23*2 # number of protons
        # Total cross section
        if flavor == 'nu_e':
            total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_O16(x)*x*1e-38 + cs_nc_nue_O16(x)*x*1e-38
        elif flavor == 'nubar_e':
            total_cross = cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_O16(x)*x*1e-38 + cs_nc_nuebar_O16(x)*x*1e-38
        elif flavor == 'nu_x':
            total_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38
        else:
            raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')
        # Detector efficiency
        eff = efficiency_sigmoid(x, 1, 7, 3.5)
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
        n_target = 6.03e32
        # Total cross section
        if flavor == 'nu_e':
            total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_Ar40(x)*x*1e-38 + cs_nc_nue_Ar40(x)*x*1e-38
        elif flavor == 'nubar_e':
            total_cross = cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_Ar40(x)*x*1e-38 + cs_nc_nuebar_Ar40(x)*x*1e-38
        elif flavor == 'nu_x':
            total_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38 + cs_nc_nux_Ar40(x)*x*1e-38 + cs_nc_nuxbar_Ar40(x)*x*1e-38
        else:
            raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')
        # Detector efficiency
        eff = efficiency_sigmoid(x, 0.98, 1.2127, 8.0591)
    elif detector == 'JUNO':
        channels = ['ibd','nue_e','nuebar_e','nue_C12','nuebar_C12','numu_e',
                    'numubar_e','nc_nue_C12','nc_nuebar_C12','nc_numu_C12',
                    'nc_numubar_C12']
        xs_data = cs.snowglobes(channels)

        # All these cross sections are in units of 10⁻³⁸ cm²/MeV
        cs_ibd = interp1d(1e3*(10**(xs_data[0][0])), xs_data[0][4]/1e3, fill_value = (0,xs_data[0][1][-1]/1e3))
        cs_nue_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[1][1]/1e3, fill_value = (0,xs_data[1][1][-1]/1e3))
        cs_nuebar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[2][4]/1e3, fill_value = (0,xs_data[2][4][-1]/1e3))
        cs_nue_C12 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[3][1]/1e3, fill_value = (0,xs_data[3][1][-1]/1e3))
        cs_nuebar_C12 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[4][1]/1e3, fill_value = (0,xs_data[4][1][-1]/1e3))
        cs_nux_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[5][2]/1e3, fill_value = (0,xs_data[5][2][-1]/1e3))
        cs_nuxbar_e = interp1d(1e3*(10**(xs_data[0][0])), xs_data[6][5]/1e3, fill_value = (0,xs_data[6][5][-1]/1e3))
        cs_nc_nue_C12 = interp1d(1e3*(10**(xs_data[7][0])), xs_data[7][1]/1e3, fill_value = 'extrapolate')
        cs_nc_nuebar_C12 = interp1d(1e3*(10**(xs_data[8][0])), xs_data[8][4]/1e3, fill_value = 'extrapolate')
        cs_nc_nux_C12 = interp1d(1e3*(10**(xs_data[9][0])), xs_data[9][2]/1e3, fill_value = 'extrapolate')
        cs_nc_nuxbar_C12 = interp1d(1e3*(10**(xs_data[10][0])), xs_data[10][5]/1e3, fill_value = 'extrapolate')
        # Number of target particles
        """"
        Fengpeng, A., Guangpeng, A. N., Qi, A. N., Antonelli, V., Baussan, E.,
        Beacom, J., ... & Sinev, V. (2016). Neutrino physics with JUNO.
        """
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
        eff = np.where(x<1,0,1) #efficiency_sigmoid(x, 0.918, 1.2127, 3)
    elif detector == 'Hyper-k':
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
        n_target = (216e9/18.01528)*6.022e23*2 # number of protons
        # Total cross section
        if flavor == 'nu_e':
            total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_O16(x)*x*1e-38 + cs_nc_nue_O16(x)*x*1e-38
        elif flavor == 'nubar_e':
            total_cross = cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_O16(x)*x*1e-38
        elif flavor == 'nu_x':
            total_cross = cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38
        else:
            raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')
        # Detector efficiency
        eff = np.where(x<3,0,0.9) #efficiency_sigmoid(x, 0.918, 1.2127, 3)
    else:
        raise ValueError("Ops, we don't have this detector in our simulations. Try 'super-k', \
            'Hyper-k', 'DUNE' or 'JUNO'.")
    # Normalization
    distance_cm = distance*3.086e21 #conversion from kiloparsec to centimeter
    A = n_target/(4 * np.pi * distance_cm**2)
    # Emission spectrum
    spectrum = emitted_spectrum(x, flavor, E_tot, hierarchy, phi)
    return A*spectrum*total_cross*eff

# Sampling from the expected detection spectra
def energy_sampler(E, E_tot, resolution, resolution_function = 'constant',
                    detector = 'super-k', hierarchy = 'normal', distance = 10,
                    print_expected = True, only_total_events = False, phi = 0):
    """"
    This function samples individual detected energies for the events based on the
    total number of expected events and the energy spectrum. It applies a random fluctuation
    to the total number of detected events N and another to each individual energy sampled in
    order to simulate detector error.

    E - neutrino energies

    E_tot - Total energy

    resolution - Detector resolution when reconstructing the energy of a neutrino

    For the DUNE detector, resolution is about 10% of E. Therefore, when using "DUNE" as
    detector, set resolution = 0.1 and resolution function = "proportional"
    For reference see: Abi, B. et al. (2021). The European Physical Journal C, 81(5), 1-26.

    resolution function - Either constant or linearly proportional to the detection energy

    detector - detector name

    hierarchy - neutrino mass hierarchy, either "normal" or "inverted"
    """
    # Calculating the number of expected neutrinos from each flavor
    N_expected_e = np.round(simps(detection_spectra(E, E_tot, 'nu_e', detector, hierarchy, distance, phi), E), 0)
    N_expected_ebar = np.round(simps(detection_spectra(E, E_tot, 'nubar_e', detector, hierarchy, distance, phi), E), 0)
    N_expected_x = np.round(simps(detection_spectra(E, E_tot, 'nu_x', detector, hierarchy, distance, phi), E), 0)

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
    weights_e = detection_spectra(E, E_tot, 'nu_e')/N_expected_e
    weights_ebar = detection_spectra(E, E_tot, 'nubar_e')/N_expected_ebar
    weights_x = detection_spectra(E, E_tot, 'nu_x')/N_expected_x

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

        # angles = np.linspace(0, np.pi/2, 10000)
        # N = simps(cs.dsigma_dtheta(angles, 1, sigma_0), angles)

        final_thetas_e = []
        final_phis_e = []
        final_thetas_ebar = []
        final_phis_ebar = []
        final_thetas_x = []
        final_phis_x = []

        # Electron neutrinos
        for i in range(len(samples_e)):
            #if elastic_scattering
            #weights = cs.dsigma_dtheta(angles, samples_e[i], sigma_0)/N
            #theta = choices(angles, weights, k = 1)
            #phi = choices(angles, weights, k = 1)
            #new_theta = np.random.normal(theta[0], np.pi/9)
            #new_phi = np.random.normal(phi[0], np.pi/9)
            #final_thetas_e.append(new_theta)
            #final_phis_e.append(new_phi)
            #else
            #theta_2 = np.random.uniform(-np.pi, np.pi)
            #phi_2 = np.random.uniform(-np.pi/2, np.pi/2)
            if resolution_function == 'constant':
                new_sample_e = np.random.normal(samples_e[i], resolution)
            elif resolution_function == 'proportional':
                new_sample_e = np.random.normal(samples_e[i], samples_e[i]*resolution)
            final_samples_e.append(new_sample_e)
        # Electron antineutrinos
        for i in range(len(samples_ebar)):
            if resolution_function == 'constant':
                new_sample_ebar = np.random.normal(samples_ebar[i], resolution)
            elif resolution_function == 'proportional':
                new_sample_ebar = np.random.normal(samples_ebar[i], samples_ebar[i]*resolution)
            final_samples_ebar.append(new_sample_ebar)
        # Mu/Tau (anti)neutrinos
        for i in range(len(samples_x)):
            if resolution_function == 'constant':
                new_sample_x = np.random.normal(samples_x[i], resolution)
            elif resolution_function == 'proportional':
                new_sample_x = np.random.normal(samples_x[i], samples_x[i]*resolution)
            final_samples_x.append(new_sample_x)
        
        final_samples_e = np.array(final_samples_e)
        final_samples_ebar = np.array(final_samples_ebar)
        final_samples_x = np.array(final_samples_x)

        # Organizing outputs
        final_samples = {}
        final_samples['nu_e'] = final_samples_e
        final_samples['nubar_e'] = final_samples_ebar
        final_samples['nu_x'] = final_samples_x
        total_sample = np.concatenate((final_samples_e,final_samples_ebar,final_samples_x))
        final_samples['Total'] = total_sample
        return final_samples