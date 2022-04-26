import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from detection import *

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]})
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['font.family'] = 'STIXGeneral'

E = np.linspace(1.804, 100, 1000000) # Neutrino detection energy
distance = 10000 # Parsecs
detector = 'super-k'
resolution = 0.05 # Fraction of error in energy reconstruction

ref = energy_sampler(E, 2e53, resolution, 'proportional', detector, 'normal', distance)
ref_invert = energy_sampler(E, 2e53, resolution, 'proportional', detector, 'inverted', distance)

fig = plt.figure(figsize=(10,6))
plt.hist(ref['Total'], bins = [2*i for i in range(50)], histtype = 'step', color = 'black',
        lw = 2, label = r'$\nu_e + \overline{\nu}_e + \nu_x$ (NO)')
plt.hist(ref_invert['Total'], bins = [2*i for i in range(50)], histtype = 'step', color = 'black',
        lw = 2, label = r'$\nu_e + \overline{\nu}_e + \nu_x$ (IO)', ls = '--')
# plt.hist(ref['nu_e'], bins = [2*i for i in range(50)], histtype = 'step', color = 'crimson',
#         lw = 2, label = r'$\nu_e$')
# plt.hist(ref['nubar_e'], bins = [2*i for i in range(50)], histtype = 'step', color = 'teal',
#         lw = 2, label = r'$\overline{\nu}_e$')
# plt.hist(ref['nu_x'], bins = [2*i for i in range(50)], histtype = 'step', color = 'orange',
#         lw = 2, label = r'$\nu_x$')
lgd = plt.legend(title = 'Flavors', loc = 'upper right', fontsize = 12)
title = lgd.get_title()
title.set_fontsize(14)
plt.xlabel('Neutrino energy [MeV]', fontsize = 14)
plt.ylabel('Events', fontsize = 14)
plt.title(f'Detection spectrum {detector} for {int(distance/1000)} Kpc Supernova', fontsize = 18)
plt.xlim(0, 70)
# plt.savefig(f'detection_spectrum_{detector}_{int(distance/1000)}_kpc.png',
#             bbox_inches = 'tight', dpi = 300)
plt.show()