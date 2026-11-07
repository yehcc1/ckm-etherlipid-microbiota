"""Figure S8 — community diversity, and whether the panel is richness — 20260913_v60.

Four panels, every number read from the files written by X5e/X5f:

  A  alpha diversity across CKM stages 0-4 (the calibrated negative)
  B  principal coordinates of the Aitchison distance, coloured by CKM stage
  C  PERMANOVA variance explained by each term
  D  the ether-lipid ratio on composition and on richness, separately and together
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy import stats
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
ST = '20260913_v60'
UP = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'

def sci(p, d=1):
    if p >= 0.01: return f'{p:.3g}'
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f'{m:.{d}f} × 10$^{{{e}}}$'
def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
def pan(ax, l, x=-0.20, y=1.04):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)

A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = (A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
        (~A0.subject_id.str.startswith('J'))]
     .drop_duplicates('subject_id').set_index('subject_id'))
from stage_v10 import master_stage
A['CKM5'] = master_stage(A, 'X5h MB825')
AL = pd.read_csv(f'{HUB}/alpha_diversity.csv').set_index('sample_id').reindex(A.sample_id.values)
AL.index = A.index
KW = pd.read_csv(f'TableS26_alpha_diversity_{ST}.csv')
PM = pd.read_csv(f'TableS27_permanova_{ST}.csv')
FO = pd.read_csv(f'TableS28_panel_vs_alpha_{ST}.csv')
Z = np.load(f'beta_pcoa_{ST}.npz', allow_pickle=True)
n825 = len(A)

fig = plt.figure(figsize=(7.2, 8.6))
gs = fig.add_gridspec(3, 2, height_ratios=[0.86, 1.0, 0.92], hspace=0.42, wspace=0.30,
                      left=0.10, right=0.975, top=0.945, bottom=0.06)

# ------------------------------------------------------------------ A
gsa = gs[0, :].subgridspec(1, 3, wspace=0.34)
STAGES = list(range(5))
RAMP = ['#CFE3F5', '#9DC6EB', '#5B9BD5', '#2E75B6', '#1F4E79']
for j, (nm, col) in enumerate([('Observed ASVs', AL.n_asv), ('Shannon', AL.shannon),
                               ('Pielou evenness', AL.evenness)]):
    ax = fig.add_subplot(gsa[0, j])
    if j == 0: pan(ax, 'A', x=-0.30)
    data = [col[A.CKM5 == s].dropna().values for s in STAGES]
    bp = ax.boxplot(data, positions=STAGES, widths=0.62, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.0),
                    whiskerprops=dict(color=GRAY, lw=0.7), capprops=dict(color=GRAY, lw=0.7))
    for patch, c in zip(bp['boxes'], RAMP):
        patch.set(facecolor=c, edgecolor=GRAY, lw=0.7)
    p = float(KW[(KW.metric == nm) & (KW.test.str.startswith('Kruskal'))].p.iloc[0])
    ax.set_title(f'{nm}\nKruskal–Wallis $P$ = {p:.2f}', fontsize=7.2, color=INK, pad=4)
    ax.set_xlabel('CKM stage', fontsize=7); ax.set_xticks(STAGES)
    if j == 0: ax.set_ylabel(f'value (n = {n825})', fontsize=7)
    clean(ax)

# ------------------------------------------------------------------ B
b = fig.add_subplot(gs[1, 0]); pan(b, 'B')
pc = Z['Aitchison (CLR, prevalent genera)_pc']; var = Z['Aitchison (CLR, prevalent genera)_var']
stage = Z['stage']
for s in STAGES:
    m = stage == s
    b.scatter(pc[m, 0], pc[m, 1], s=9, color=RAMP[s], edgecolors='none', alpha=0.85,
              label=f'{s} (n = {int(m.sum())})', rasterized=True)
b.axhline(0, color=GRID, lw=0.6); b.axvline(0, color=GRID, lw=0.6)
b.set_xlabel(f'PCo1 ({100*var[0]:.1f}%)'); b.set_ylabel(f'PCo2 ({100*var[1]:.1f}%)')
row = PM[(PM.distance.str.startswith('Aitchison')) & (PM.model == 'univariable') &
         (PM.term == 'CKM stage')].iloc[0]
b.set_title(f'Aitchison distance, by CKM stage\nPERMANOVA $R^2$ = {100*row.R2:.1f}%, '
            f'$P$ = {row.p:.3f}', fontsize=7.6, color=INK, pad=4)
lg = b.legend(title='CKM stage', fontsize=5.6, title_fontsize=6, loc='upper left',
              frameon=True, facecolor='white', edgecolor='none', framealpha=0.82,
              handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25)
lg._legend_box.align = 'left'
clean(b)

# ------------------------------------------------------------------ C
c = fig.add_subplot(gs[1, 1]); pan(c, 'C', x=-0.34)
u = PM[(PM.distance.str.startswith('Aitchison')) & (PM.model == 'univariable')].copy()
u = u.set_index('term').loc[['16S batch', 'Age', 'Sex', 'Body-mass index', 'CKM stage',
                             'Ether-lipid ratio', 'Microbiota panel']].reset_index()
y = np.arange(len(u))[::-1]
cols = [GRAY if t == 'Microbiota panel' else BLUE for t in u.term]
c.barh(y, 100 * u.R2, 0.62, color=cols, edgecolor='white', lw=0.6)
for yy, (_, r) in zip(y, u.iterrows()):
    c.text(100 * r.R2 + 0.12, yy, f'$P$ = {r.p:.3f}', va='center', fontsize=5.8, color=MUTED)
c.set_yticks(y); c.set_yticklabels(u.term, fontsize=6.4)
c.set_xlabel('Variance in community composition explained (%)')
c.set_xlim(0, 10.6)
c.set_title('PERMANOVA, each term alone', fontsize=7.6, color=INK, pad=4)
c.text(0.99, 0.40, 'grey: the panel is derived from the\nsame genus table, so this is a\nprojection of it, not a test',
       transform=c.transAxes, ha='right', va='bottom', fontsize=5.6, color=MUTED,
       style='italic', linespacing=1.4)
clean(c)

# ------------------------------------------------------------------ D
d = fig.add_subplot(gs[2, :]); pan(d, 'D', x=-0.093)
ROWS = [('CV panel + depth', 'CV panel', 'alone'),
        ('CV panel + Observed ASVs + depth', 'CV panel', 'with observed ASVs in the model'),
        ('CV panel + Shannon + depth', 'CV panel', 'with Shannon in the model'),
        ('Pre-specified score + depth', 'Pre-specified score', 'alone'),
        ('Pre-specified score + Observed ASVs + depth', 'Pre-specified score',
         'with observed ASVs in the model'),
        ('Pre-specified score + Shannon + depth', 'Pre-specified score',
         'with Shannon in the model'),
        ('Observed ASVs + depth', 'Observed ASVs', 'alone'),
        ('Shannon + depth', 'Shannon', 'alone')]
vals = []
for m, t, lab in ROWS:
    r = FO[(FO.model == m) & (FO.term == t)].iloc[0]
    vals.append((f'{t} — {lab}', r.beta, r.se, r.p, int(r.n), t))
y = np.arange(len(vals))[::-1]
for yy, (lab, beta, se, p, n, t) in zip(y, vals):
    col = BLUE if t in ('CV panel', 'Pre-specified score') else PURPLE
    d.errorbar(beta, yy, xerr=1.96 * se, fmt='o', ms=4.2, lw=1.1, color=col, ecolor=col, capsize=2)
    d.text(beta + 1.96 * se + 0.010, yy, f'$P$ = {sci(p)}', va='center', fontsize=5.9, color=INK)
d.axvline(0, color=GRAY, lw=0.8, ls='--')
d.set_yticks(y); d.set_yticklabels([v[0] for v in vals], fontsize=6.3)
d.set_xlabel('Effect on the plasma ether-lipid ratio (SD per SD)')
d.set_xlim(-0.02, 0.36)
d.set_title(f'Composition and richness, separately and together (n = {vals[0][4]})',
            fontsize=7.6, color=INK, pad=4)
d.text(0.5, -0.30, 'every model adjusted for age, sex, body-mass index, 16S sequencing batch '
                   'and log read depth', transform=d.transAxes, ha='center', va='top',
       fontsize=6.0, color=MUTED)
clean(d)

plt.savefig(f'FigureS8_{ST}.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('S8_v60.png', dpi=185, bbox_inches='tight', pad_inches=0.10)
plt.close()
print('Figure S8 done')
