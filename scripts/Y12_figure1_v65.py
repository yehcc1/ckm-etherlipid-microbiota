# -*- coding: utf-8 -*-
"""Figure 1 重畫：拿掉 RNA-seq 層（論文不再提 RNA）    20261031_v65"""
"""Main figures 1-4, rebuilt on the 825/743 analysis sets — 20260911_v54.

Every value is read from the v54 result files; nothing is typed in.
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, re
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from itertools import combinations
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
UP = '/mnt/user-data/uploads/CKM+microbiome'; ST = '20261031_v65'
N = pd.read_csv('numbers_v65_all_20261030_v65.csv').drop_duplicates('key', keep='last').set_index('key').value
nv = lambda k: float(N[k]); ni = lambda k: int(float(N[k]))
def sci(p, d=1):
    if p >= 0.01: return f'{p:.3g}'
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f'{m:.{d}f} × 10$^{{{e}}}$'
def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
def pan(ax, l, x=-0.18, y=1.05):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)

# =========================== FIGURE 1 ===========================
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
LAY = {'Metabolome': set(PHE.subject_id),
       'Microbiome': set(A.subject_id)}
n_metab, n_mb = len(LAY['Metabolome']), len(LAY['Microbiome'])
n_both = len(LAY['Metabolome'] & LAY['Microbiome'])
assert n_both == ni('n_microbiome') and n_mb == ni('n_microbiome_all')

fig = plt.figure(figsize=(3.8, 3.4))
a = fig.add_axes([0.02, 0.02, 0.96, 0.96]); a.axis('off'); a.set_xlim(0, 100); a.set_ylim(0, 100)
a.text(50, 100, 'Metabolic cohort, Chang Gung Memorial Hospital, Keelung',
       ha='center', va='top', fontsize=6.6, color=INK)

def box(x, y, w, h, txt, fc, ec, lw=1.1, fs=6.2):
    a.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0.6,rounding_size=1.4',
                               fc=fc, ec=ec, lw=lw, zorder=2))
    a.text(x + w / 2, y - h / 2, txt, ha='center', va='center', fontsize=fs,
           color=INK, linespacing=1.5, zorder=3)

box(2, 93, 44, 14, f'Targeted plasma\nmetabolomics\nn = {n_metab:,}', '#EAF2FA', BLUE)
box(54, 93, 44, 14, f'Faecal 16S rRNA within\n180 days of a clinical visit\nn = {n_mb:,}',
    '#E7F5EF', GREEN)
for x0 in (24, 76):
    a.annotate('', xy=(50, 74.0), xytext=(x0, 78.0),
               arrowprops=dict(arrowstyle='-|>', lw=1.1, color=GRAY, mutation_scale=9))
box(13, 73, 74, 14, f'BOTH LAYERS   n = {n_both:,}\nprimary analysis set for every model\n'
                    f'with a metabolite outcome or mediator', '#EAF2FA', BLUE, lw=1.7)
a.text(50, 48, f'Models that need only microbiota and clinical data use all {n_mb:,}\n'
               f'profiled participants; the {n_mb - n_both} without metabolomics enter\n'
               f'no metabolite model and serve as an independent hold-out.',
       ha='center', va='center', fontsize=5.9, color=MUTED, linespacing=1.7, style='italic')
a.text(50, 30, f'Ratio–outcome models use all {ni("n_metabolite_clinical"):,} participants with\n'
               f'metabolomics and a clinical visit.',
       ha='center', va='center', fontsize=5.9, color=MUTED, linespacing=1.7, style='italic')

plt.savefig(f'Figure1_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.savefig('chk/F1_v65.png', dpi=170, bbox_inches='tight'); plt.close()
print('Figure 1 done')

