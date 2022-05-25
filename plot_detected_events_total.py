import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from detection import *
from tqdm import tqdm
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]})
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['font.family'] = 'STIXGeneral'

E = np.linspace(1.804, 100, 1000000) # Neutrino detection energy
detector = 'super-k'
resolution = 0.05 # Fraction of error in energy reconstruction

results = {}
distances = []
means_no = []
means_io = []
std_no = []
std_io = []
for i in tqdm(range(20)):
    distance = 5000 + 2000*i
    tot_no = []
    tot_io = []
    for j in range(20):
        ref = energy_sampler(E, 2e53, resolution, 'proportional', detector, 'normal', distance, print_expected=False, only_total_events=True)
        ref_invert = energy_sampler(E, 2e53, resolution, 'proportional', detector, 'inverted', distance, print_expected=False, only_total_events=True)
        tot_no.append(ref['Total'])
        tot_io.append(ref_invert['Total'])
    distances.append(distance)
    means_no.append(np.mean(tot_no))
    means_io.append(np.mean(tot_io))
    std_no.append(np.std(tot_no))
    std_io.append(np.std(tot_io))

distances = np.array(distances)
means_no = np.array(means_no)
means_io = np.array(means_io)
std_no = np.array(std_no)
std_io = np.array(std_io)

results['Distances'] = distances
results['Mean N normal'] = means_no
results['Mean N inverted'] = means_io
results['SD N normal'] = std_no
results['SD N inverted'] = std_io

pd.DataFrame(results).to_csv(f'Total_events_detected_{detector}.csv')

plt.plot(distances, means_no)
plt.plot(distances, means_io)
plt.yscale('log')
plt.show()