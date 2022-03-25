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
def efficiency_super_k(x, a, b, c):
    """
    100% em 4.5 MeV e 97% em 4.0 MeV
    Fukuda, S., Fukuda, Y., Hayakawa, T., Ichihara, E., Ishitsuka, M., Itow, Y., ... & Ichikawa, Y. (2003). The super-kamiokande detector.
    Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 501(2-3), 418-462.
    """
    eff = a/(1+np.exp(-b*(x-c)))
    return eff

E = np.linspace(1.804, 100, 1000000) # Neutrino detection energy
distance = 10000 # Parsecs

def detection_spectra(x, E_tot, flavor = 'nu_e', detector = 'super-k', hierarchy = 'normal'):
    if detector == 'super-k':
        # Number of target particles
        n_target = (32000000000/18.01528)*6.022e23*2
        # Individual cross sections
        cross_oxygen = cs.cross_section_nu_e_oxygen(x)
        cross_oxygen_anti = cs.cross_section_nubar_e_oxygen(x)
        cross_scatter = cs.cross_section_NC_nu_e(x, 5, g1_nu_e, g2_nu_e)
        cross_scatter_anti = cs.cross_section_NC_nu_e(x, 5, g1_barnu_e, g2_barnu_e)
        cross_scatter_x = cs.cross_section_NC_nu_e(x, 5, g1_nu_x, g2_nu_x)
        cross_scatter_anti_x = cs.cross_section_NC_nu_e(x, 5, g1_barnu_x, g2_barnu_x)
        cross_ivb = cs.cross_section_CC_nu_proton(x)
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
        eff = efficiency_super_k(x, 100, 7, 3.5)
    else:
        raise ValueError("Ops, we don't have this detector in our simulations. Try 'super-k'.")
    # Normalization
    distance_cm = distance*3.09e16
    A = n_target/(4 * np.pi * distance_cm**2)
    # Emission spectrum
    spectrum = emitted_spectrum(x, flavor, E_tot, hierarchy)
    return A*spectrum*total_cross*eff

# Sampling from the expected detection spectra
def energy_sampler(E, E_tot, resolution, resolution_function = 'constant'):
    # Calculating the number of expected neutrinos from each flavor
    N_expected_e = np.round(simps(detection_spectra(E, E_tot), E), 0)
    N_expected_ebar = np.round(simps(detection_spectra(E, E_tot, 'nubar_e'), E), 0)
    N_expected_x = np.round(simps(detection_spectra(E, E_tot, 'nu_x'), E), 0)

    print(f'Number of expected neutrinos by flavor at {distance} parsecs'
    '\n'
    '\n'
    f'electron neutrinos: {int(N_expected_e, 0)}'
    '\n'
    f'electron antineutrinos: {int(N_expected_ebar, 0)}'
    '\n'
    f'mu/tau (anti)neutrinos: {int(N_expected_x, 0)}')
    
    # Sampling
    possible_energies = E
    weights_e = detection_spectra(E, E_tot, 'nu_e')/N_expected_e
    weights_ebar = detection_spectra(E, E_tot, 'nubar_e')/N_expected_ebar
    weights_x = detection_spectra(E, E_tot, 'nu_x')/N_expected_x

    # Sampling process uses poisson distribution to resample the number of detected neutrinos in each flavor
    samples_e = choices(possible_energies, weights_e, k = np.random.poisson(int(N_expected_e)))
    samples_ebar = choices(possible_energies, weights_ebar, k = np.random.poisson(int(N_expected_ebar)))
    samples_x = choices(possible_energies, weights_x, k = np.random.poisson(int(N_expected_x, 0)))
    
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
    individual_samples = {}
    individual_samples['nu_e'] = final_samples_e
    individual_samples['nubar_e'] = final_samples_ebar
    individual_samples['nu_x'] = final_samples_x
    total_sample = final_samples_e + final_samples_ebar + final_samples_x
    return total_sample, individual_samples