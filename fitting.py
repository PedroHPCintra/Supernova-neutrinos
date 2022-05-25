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

@njit
def spectrum_shape(x, alpha, E_mean, E_tot):
    """
    Overral shape of the emission spectrum
    """
    A = E_tot*((1+alpha)**(1+alpha))/((gamma(1+alpha))*E_mean**(2+alpha))
    f_nu = A*(x**alpha)*np.exp(-(alpha + 1)*x/E_mean)
    return f_nu

@njit
def fitting_spectra(x, alpha, E_mean, E_tot, detector = 'super-k'):
    """
    Shape of the detection spectrum
    """
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
        total_cross = cross_oxygen_anti + cross_scatter_anti + cross_ivb + cross_oxygen + cross_scatter + cross_scatter_x + cross_scatter_anti_x
        # Detector efficiency
        eff = dtc.efficiency_sigmoid(x, 100, 7, 3.5)
    elif detector == 'DUNE':
        # Number of target particles
        n_target = 6.03e32
        # Total cross section
        total_cross = cross_scatter + cross_argon + cross_scatter_anti + cross_ivb + cross_argon_anti + cross_scatter_x + cross_scatter_anti_x
        # Detector efficiency
        eff = dtc.efficiency_sigmoid(x, 98, 1.2127, 8.0591)
    else:
        raise ValueError("Ops, we don't have this detector in our simulations. Try 'super-k' \
            or 'DUNE'.")

    # Combined spectrum
    spectrum = spectrum_shape(x, alpha, E_mean, E_tot)
    all = spectrum*total_cross*eff
    # Normalization
    A = simps(all, x)
    return all/A

@njit
def rejABC(model, x, y, prior_params, eps, n_sample):
    """
    This is the fitting algorithm, a rejection ABC.
    model: function to be fitted
    x: x data
    y: y data
    prior_params: list of ranges for uniform priors for all free parameters
    eps: selected tolerance
    n_sample: number of samples to be sorted
    """
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
    
    return post[1:]

@njit
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

@njit
def smcABC(model, x, y, hist, bins, n_bins, p_std, eps, n_sample, n_max):
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
            
    return post[1:]