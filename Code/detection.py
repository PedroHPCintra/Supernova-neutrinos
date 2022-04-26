import numpy as np
import scipy.stats as st
from scipy.integrate import simps
import pandas as pd
from random import choices
from math import gamma
import cross_sections as cs
from constants import *
from emission import *

# Detector efficiency
def efficiency_sigmoid(x, a, b, c):
    """
    100% em 4.5 MeV e 97% em 4.0 MeV
    Fukuda, S., Fukuda, Y., Hayakawa, T., Ichihara, E., Ishitsuka, M., Itow, Y., ... & Ichikawa, Y. (2003). The super-kamiokande detector.
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 501(2-3), 418-462.
    """
    eff = a/(1+np.exp(-b*(x-c)))
    return eff

E = np.linspace(1.804, 100, 1000000) # Neutrino detection energy

def detection_spectra(x, E_tot, flavor = 'nu_e', detector = 'super-k', hierarchy = 'normal', distance = 10000):
    cross_oxygen = cs.cross_section_nu_e_oxygen(x)
    cross_oxygen_anti = cs.cross_section_nubar_e_oxygen(x)
    cross_scatter = cs.cross_section_NC_nu_e(x, 5, g1_nu_e, g2_nu_e)
    cross_scatter_anti = cs.cross_section_NC_nu_e(x, 5, g1_barnu_e, g2_barnu_e)
    cross_scatter_x = cs.cross_section_NC_nu_e(x, 5, g1_nu_x, g2_nu_x)
    cross_scatter_anti_x = cs.cross_section_NC_nu_e(x, 5, g1_barnu_x, g2_barnu_x)
    cross_ivb = cs.cross_section_CC_nu_proton(x)
    cross_argon = cs.cross_section_nu_e_argon(x)
    cross_argon_anti = cs.cross_section_nubar_e_argon(x)
    if detector == 'super-k':
        # Number of target particles
        n_target = (32000000000/18.01528)*6.022e23*2
        # Total cross section
        if flavor == 'nu_e':
            total_cross = cross_oxygen + cross_scatter
        elif flavor == 'nubar_e':
            total_cross = cross_oxygen_anti + cross_scatter_anti + cross_ivb
        elif flavor == 'nu_x':
            total_cross = cross_scatter_x + cross_scatter_anti_x
        else:
            raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')
        # Detector efficiency
        eff = efficiency_sigmoid(x, 1, 7, 3.5)
    elif detector == 'DUNE':
        # Number of target particles
        n_target = 6.03e32
        # Total cross section
        if flavor == 'nu_e':
            total_cross = cross_scatter + cross_argon
        elif flavor == 'nubar_e':
            total_cross = cross_scatter_anti + cross_ivb + cross_argon_anti
        elif flavor == 'nu_x':
            total_cross = cross_scatter_x + cross_scatter_anti_x
        else:
            raise ValueError('Invalid entry for neutrino species. Please use "nu_e" for \
                electron neutrinos, "nubar_e" for electron antineutrinos or "nu_x" for mu or \
                tau (anti)neutrinos')
        # Detector efficiency
        eff = efficiency_sigmoid(x, 0.98, 1.2127, 8.0591)
    else:
        raise ValueError("Ops, we don't have this detector in our simulations. Try 'super-k' \
            or 'DUNE'.")
    # Normalization
    distance_cm = distance*3.09e16
    A = n_target/(4 * np.pi * distance_cm**2)
    # Emission spectrum
    spectrum = emitted_spectrum(x, flavor, E_tot, hierarchy)
    return A*spectrum*total_cross*eff

# Sampling from the expected detection spectra
def energy_sampler(E, E_tot, resolution, resolution_function = 'constant',
                    detector = 'super-k', hierarchy = 'normal', distance = 10000,
                    print_expected = True, only_total_events = False):
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

    detector - detector type

    hierarchy - neutrino mass hierarchy, either "normal" or "inverted"
    """
    # Calculating the number of expected neutrinos from each flavor
    N_expected_e = np.round(simps(detection_spectra(E, E_tot, 'nu_e', detector, hierarchy, distance), E), 0)
    N_expected_ebar = np.round(simps(detection_spectra(E, E_tot, 'nubar_e', detector, hierarchy, distance), E), 0)
    N_expected_x = np.round(simps(detection_spectra(E, E_tot, 'nu_x', detector, hierarchy, distance), E), 0)

    if print_expected:
        print('\n'
        f'Number of expected neutrinos by flavor at {distance} parsecs'
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
        
        ##### Noise in detection energy #####
        final_samples_e = []
        final_samples_ebar = []
        final_samples_x = []
        # Electron neutrinos
        for i in range(len(samples_e)):
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