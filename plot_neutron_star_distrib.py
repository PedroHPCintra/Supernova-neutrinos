import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

ns = pd.read_csv('https://raw.githubusercontent.com/PedroHPCintra/Supernova-neutrinos/main/Data/Neutron_star_mass_distribution.csv')

galactic_ns = ns.loc[ns['Source'] == 'Galactic']
ns_ns = galactic_ns.loc[galactic_ns['Type'] == 'NS-NS']
ns_wd = galactic_ns.loc[galactic_ns['Type'] == 'NS-WD']

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]})
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['font.family'] = 'STIXGeneral'

fig = plt.figure(figsize=(10,8))
sns.kdeplot(data = ns, x = 'Mass', hue = 'Type', fill=True, legend=False)
plt.xlim(0, 5.5)
plt.ylim(0, 1.5)
plt.fill_between([5,6], 0, 2, color = 'grey', alpha = 0.6)
plt.vlines(5, 0, 1.7, linestyles='dashed', color = 'black')
plt.text(5.2, 0.75, 'Stellar black hole', fontsize = 14, rotation = 'vertical',
        ha = 'center', rotation_mode = 'anchor')
plt.text(1.42, 1.22, 'NS-NS binary', fontsize = 12)
plt.text(1.92, 0.26, 'NS-WD binary', fontsize = 12)
plt.text(2.15, 0.1, 'HMXB', fontsize = 12)
plt.xlabel('Mass $M_\odot$', fontsize = 14)
plt.ylabel('Density', fontsize = 14)
plt.tight_layout()
fig.patch.set_alpha(1)
for ax in fig.axes:
        ax.patch.set_alpha(1)
plt.savefig('neutron_star_mass_distribution.png', bbox_inches = 'tight', dpi = 300)
plt.show()