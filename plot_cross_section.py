import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cross_sections as cs
from constants import g1_nu_e, g2_nu_e, g1_barnu_e, g2_barnu_e, g1_nu_x, g2_nu_x, g1_barnu_x, g2_barnu_x

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\usepackage{underscore}')
# mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]})
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['font.family'] = 'STIXGeneral'

x = np.linspace(10, 100, 1000)

cross_oxygen = cs.cross_section_nu_e_oxygen(x)
cross_oxygen_anti = cs.cross_section_nubar_e_oxygen(x)
cross_scatter = cs.cross_section_NC_nu_e(x, 5, g1_nu_e, g2_nu_e)
cross_scatter_anti = cs.cross_section_NC_nu_e(x, 5, g1_barnu_e, g2_barnu_e)
cross_scatter_x = cs.cross_section_NC_nu_e(x, 5, g1_nu_x, g2_nu_x)
cross_scatter_anti_x = cs.cross_section_NC_nu_e(x, 5, g1_barnu_x, g2_barnu_x)
cross_ivb = cs.cross_section_CC_nu_proton(x)
cross_argon = cs.cross_section_nu_e_argon(x)
cross_argon_anti = cs.cross_section_nubar_e_argon(x)

fig = plt.figure(figsize=(10,8))
plt.plot(x, cross_oxygen, color = 'crimson', lw = 2,
        label = r'$\nu_e + ^{16}\mathrm{O} \rightarrow \mathrm{X} + e^-$')
plt.plot(x, cross_oxygen_anti, color = 'crimson', lw = 2, ls = '--',
        label = r'$\overline{\nu}_e + ^{16}\mathrm{O} \rightarrow \mathrm{X} + e^+$')
plt.plot(x, cross_scatter, color = 'teal', lw = 2,
        label = r'$\nu_e + e^- \rightarrow \nu_e + e^-$')
plt.plot(x, cross_scatter_anti, color = 'teal', lw = 2, ls = '--',
        label = r'$\overline{\nu}_e + e^- \rightarrow \overline{\nu}_e + e^-$')
plt.plot(x, cross_scatter_x, color = 'teal', lw = 2, ls = 'dotted',
        label = r'$\nu_x + e^- \rightarrow \nu_x + e^-$')
plt.plot(x, cross_scatter_anti_x, color = 'teal', lw = 2, ls = '-.',
        label = r'$\overline{\nu}_x + e^- \rightarrow \overline{\nu}_x + e^-$')
plt.plot(x, cross_ivb, color = 'indigo', lw = 2,
        label = r'$\overline{\nu}_e + p \rightarrow n + e^+$')
plt.plot(x, cross_argon, color = 'forestgreen', lw = 2,
        label = r'$\nu_e + ^{40}\mathrm{Ar} \rightarrow ^{40}\mathrm{K}^* + e^-$')
plt.plot(x, cross_argon_anti, color = 'forestgreen', lw = 2, ls = '--',
        label = r'$\overline{\nu}_e + ^{40}\mathrm{Ar} \rightarrow ^{40}\mathrm{K}^* + e^+$')
plt.yscale('log')
lgd = plt.legend(title = 'Interactions', loc='lower right', ncol = 3, bbox_to_anchor = (1,1),
        fontsize = 12)
title = lgd.get_title()
title.set_fontsize(14)
plt.xlabel('Neutrino energy [MeV]', fontsize = 14)
plt.ylabel(r'Cross section [cm$^{2}$]', fontsize = 14)
plt.xlim(10, 100)
plt.ylim(1e-46, 1e-38)
plt.tight_layout()
fig.patch.set_alpha(0.3)
for ax in fig.axes:
        ax.patch.set_alpha(0.5)
plt.savefig('cross_sections_neutrinos.png', bbox_inches = 'tight', dpi = 300)
plt.show()