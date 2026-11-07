"""Figure 4 — metabolite-level route map (the counterpart of Figure 4c of Wu et al. 2025) — 20261031_v65.

Drawing lives in pathmap_v65.py; every statistic is read from TableSmc_metacyc_taxa_20261031_v65.csv and the
host numbers from the v65 numbers files.  Output Figure4_pathmap_20261031_v65.svg.
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5})
import pathmap_v65 as pm
N = pd.concat([pd.read_csv('numbers_v65_all_20261030_v65.csv'), pd.read_csv('numbers_extra_20261030_v65.csv')]).drop_duplicates('key', keep='last').set_index('key').value
def sci(p, d=1):
    if p >= 0.01: return f'{p:.3g}'
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f'{m:.{d}f} × 10$^{{{e}}}$'
host = dict(beta=float(N['pwy5971_ether-lipid_ratio_beta']), p=sci(float(N['pwy5971_ether-lipid_ratio_p'])), n=int(float(N['n_microbiome'])))
fig, ax = plt.subplots(figsize=(7.4, 8.4))
fig.subplots_adjust(left=0.005, right=0.995, top=0.975, bottom=0.005)
m = pm.draw(ax, 'TableSmc_metacyc_taxa_20261031_v65.csv', host)
ST = '20261031_v65'
plt.savefig(f'Figure4_pathmap_{ST}.svg', bbox_inches='tight', pad_inches=0.08)
plt.savefig('chk/F4map_v65.png', dpi=200, bbox_inches='tight', pad_inches=0.08)
print('regions', len(m.used), m.used)
