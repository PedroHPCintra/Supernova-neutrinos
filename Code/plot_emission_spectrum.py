import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from emission import *

# Neutrino energies [MeV]
E = np.linspace(0, 60, 1000)

#Function for plotting the spectra
def plot_spectrum(E, L = 2e53, stage = 'emission', hierarchy = 'normal',
                yscale = 'linear'):
    fig = plt.figure(figsize=(10,8))
    if stage == 'production':
        plt.plot(E, produced_spectrum(E, 'nu_e', L), color = 'crimson',
                label = r'$\nu_e$')
        plt.plot(E, produced_spectrum(E, 'nubar_e', L), color = 'teal',
                ls = '--', label = r'$\overline{\nu}_e$')
        plt.plot(E, produced_spectrum(E, 'nu_x', L), color = 'indigo',
                ls = '-.', label = r'$\nu_x$')
    elif stage == 'emission':
        plt.plot(E, emitted_spectrum(E, 'nu_e', L, hierarchy), color = 'crimson',
                label = r'$\nu_e$')
        plt.plot(E, emitted_spectrum(E, 'nubar_e', L, hierarchy), color = 'teal',
                ls = '--', label = r'$\overline{\nu}_e$')
        plt.plot(E, emitted_spectrum(E, 'nu_x', L, hierarchy), color = 'indigo',
                ls = '-.', label = r'$\nu_x$')
    elif stage == 'both':
        plt.plot(E, produced_spectrum(E, 'nu_e', L), color = 'crimson',
                label = r'Produced $\nu_e$')
        plt.plot(E, produced_spectrum(E, 'nubar_e', L), color = 'teal',
                label = r'Produced $\overline{\nu}_e$')
        plt.plot(E, produced_spectrum(E, 'nu_x', L), color = 'indigo',
                label = r'Produced $\nu_x$')
        plt.plot(E, emitted_spectrum(E, 'nu_e', L, hierarchy), color = 'crimson',
                ls = '--', label = r'Emitted $\nu_e$')
        plt.plot(E, emitted_spectrum(E, 'nubar_e', L, hierarchy), color = 'teal',
                ls = '--', label = r'Emitted $\overline{\nu}_e$')
        plt.plot(E, emitted_spectrum(E, 'nu_x', L, hierarchy), color = 'indigo',
                ls = '--', label = r'Emitted $\nu_x$')
    else:
        raise ValueError('Invalid stage or hierarchy. Please choose between "normal" or\
                        "inverted" for hierarchy and "production" or "emission" for stage. \
                        If you want to plot the spectra in both stages use "both".')
    plt.legend(fontsize = 12, ncol = 2, loc = 'upper right')
    plt.title('Time-integrated fluxes of supernova neutrinos in {} mass hierarchy'.format(hierarchy),
            fontsize = 18, pad = 20)
    plt.xlabel('Neutrino energy [MeV]', fontsize = 16)
    plt.ylabel('Time-integrated flux', fontsize = 16)
    plt.yscale(yscale)
    plt.xlim(0, 60)
    if yscale == 'linear':
        plt.ylim(0)
    plt.tight_layout()
    plt.show()

# Calling the function to plot spectra using normal mass hierarchy and 2e53 ergs of total luminosity
plot_spectrum(E, 2e53, 'both', 'normal', yscale='linear')