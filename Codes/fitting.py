import numpy as np
import scipy.stats as st
from scipy.integrate import simps
import pandas as pd
from random import choices
from math import gamma
import cross_sections as cs
from constants import *
from numba import njit
import detection as dtc
from scipy.interpolate import interp1d
from timeit import default_timer as timer
from datetime import timedelta

#@njit
def spectrum_shape(x, alpha, E_mean):
    """
    Overral shape of the emission spectrum
    """
    A = ((1+alpha)**(1+alpha))/((gamma(1+alpha))*E_mean**2)
    f_nu = A*((x/E_mean)**alpha)*np.exp(-(alpha + 1)*x/E_mean)
    return f_nu

#@njit
def fitting_spectra(x, alpha, E_mean, detector = 'super-k'):
    """
    Shape of the detection spectrum
    """
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
        # Total cross section
        total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_O16(x)*x*1e-38 + cs_nc_nue_O16(x)*x*1e-38 + cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_O16(x)*x*1e-38 + cs_nc_nuebar_O16(x)*x*1e-38 + cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38
        # Detector efficiency
        eff = dtc.efficiency_sigmoid(x, 1, 7, 3.5)
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
        # Total cross section
        total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_Ar40(x)*x*1e-38 + cs_nc_nue_Ar40(x)*x*1e-38 + cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_Ar40(x)*x*1e-38 + cs_nc_nuebar_Ar40(x)*x*1e-38 + cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38 + cs_nc_nux_Ar40(x)*x*1e-38 + cs_nc_nuxbar_Ar40(x)*x*1e-38
        # Detector efficiency
        eff = dtc.efficiency_sigmoid(x, 98, 1.2127, 8.0591)
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
        total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_C12(x)*x*1e-38 + cs_nc_nue_C12(x)*x*1e-38 + cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_C12(x)*x*1e-38 + cs_nc_nuebar_C12(x)*x*1e-38 + cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38 + cs_nc_nux_C12(x)*x*1e-38 + cs_nc_nuxbar_C12(x)*x*1e-38
        eff = np.where(x<1,0,1) #efficiency_sigmoid(x, 0.918, 1.2127, 3)
    elif detector == 'Hyper-k':
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
        cs_nc_nue_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[7][1]/1e3, fill_value='extrapolate')
        cs_nc_nuebar_O16 = interp1d(1e3*(10**(xs_data[0][0])), xs_data[8][4]/1e3, fill_value='extrapolate')
        # Number of target particles
        n_target = (216e9/18.01528)*6.022e23*2 # number of protons
        # Total cross section
        total_cross = cs_nue_e(x)*x*1e-38 + cs_nue_O16(x)*x*1e-38 + cs_nc_nue_O16(x)*x*1e-38 + cs_nuebar_e(x)*x*1e-38 + cs_ibd(x)*x*1e-38 + cs_nuebar_O16(x)*x*1e-38 + cs_nux_e(x)*x*1e-38 + cs_nuxbar_e(x)*x*1e-38 + cs_nc_nuebar_O16(x)*x*1e-38
        # Detector efficiency
        eff = np.where(x<3,0,0.9) #efficiency_sigmoid(x, 0.918, 1.2127, 3)
    else:
        raise ValueError("Ops, we don't have this detector in our simulations. Try 'super-k', 'JUNO', 'Hyper-k' or 'DUNE'.")

    # Combined spectrum
    spectrum = spectrum_shape(x, alpha, E_mean)
    all = spectrum*total_cross*eff
    # Normalization
    A = simps(all, x)
    return all/A

#@njit
def rejABC(model, x, y, prior_params, eps, n_sample, print_progress = True):
    """
    This is the fitting algorithm, a rejection ABC.
    model: function to be fitted
    x: x data
    y: y data
    prior_params: list of ranges for uniform priors for all free parameters
    eps: selected tolerance
    n_sample: number of samples to be sorted
    """
    start = timer()
    n_params = len(prior_params) # Number of model parameters to be fit
    p = np.zeros(n_params+1, dtype=np.float64) # Array of parameters
    post = np.zeros((1,n_params+1)) # Array to build posterior distribution

    for i in range(n_sample):
        # Sort parameters according to given priors
        for j in range(n_params):
            p[j] = np.random.uniform(prior_params[j,0], prior_params[j,1])

        d = np.sqrt(np.sum(((y-model(x, *p[:-1])))**2))/len(x) # Distance = RMSE
        p[-1] = d # Model-data distance

        # If the sorted parameters result in a distance smaller than the tolerance
        # they are appended to the posterior distribution
        if (d < eps):
            post = np.concatenate((post, p.reshape((1,n_params+1)))).reshape(len(post)+1, n_params+1)

        if print_progress:
            if i==0 or (i+1) % 500 == 0:
                end = timer()
                delta = timedelta(seconds=end-start)
                print(f'\r{100*(i+1)/n_sample:.2f}% of samples sorted. Elapsed time (h:m:s): {delta}', end="")

    return post[1:]

#@njit
def sort(n, hist, bins):
    """
    This function sorts numbers from a given histogram using the histogram as
    a probability density function
    n: number of values to be sorted
    hist: k-sized array with height of columns of the normalized histogram
    bins: (l+1)-sized array with values of bins limits
    """
    d = bins[1] - bins[0] # Bin size
    dat = [] # List of sorted random numbers

    for i in range(n):
        x = np.random.uniform(0., 1.)

        # Conversion of 0-1 random number to number sorted according to the given histogram
        for j in range(len(hist)):
            if (x < np.sum(hist[:j+1])*d):
                dat.append(np.random.uniform(bins[j], bins[j+1]))
                break

    return np.array(dat)

#@njit
def smcABC(model, x, y, hist, bins, n_bins, p_std, eps, n_sample, n_max, print_progress = True):
    """
    This function continues the fitting after the rejection ABC, it is the
    Sequencial Monte Carlo ABC.
    model: function to be fitted
    x: x data
    y: y data
    hist: heights of the histogram from the past posterior distribution
    bins: limits of bins from the histogram of the past posterior distribution
    n_bins: number of bins used to make a new prior from the last posterior
    p_std: standard deviations of the last posterior distributions, these are used
    to add noise into the sampling process
    eps: tolerance
    n_sample: number of samples to be sorted
    """
    start = timer()
    n_mp = len(hist) # Number of model parameters to be fit
    p = np.zeros(n_mp+1, dtype=np.float64) # Array of parameters
    post = np.zeros((1,n_mp+1)) # Array to build posterior distribution

    for i in range(n_sample):
        # Sort parameters according to given priors
        for j in range(n_mp):
            # p[j] = np.random.uniform(prior_params[j,0], prior_params[j,1])
            p[j] = sort(1, hist[j], bins[j]) + np.random.normal(scale=p_std[j]/n_bins)

        d = np.sqrt(np.sum(((y-model(x, *p[:-1])))**2))/len(x)
        p[-1] = d # Model-data distance

        # Check parameters and add sample to posterior distribution
        if (d < eps):
            post = np.concatenate((post, p.reshape((1,n_mp+1)))).reshape(len(post)+1, n_mp+1)

        if (len(post) > n_max):
            break

        if print_progress:
            if i==0 or (i+1) % 500 == 0:
                end = timer()
                delta = timedelta(seconds=end-start)
                print(f'\r{100*(i+1)/n_sample:.2f}% of samples sorted. Elapsed time (h:m:s): {delta}', end="")

    return post[1:]