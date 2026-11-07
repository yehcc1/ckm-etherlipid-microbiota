"""Figures 5 and 6 and supplementary Figure S7 — the analyses added in v54, 20260911.

Figure 5   how well the microbiota panel and the ether-lipid ratio actually discriminate
           clinical state, cross-validated out of fold, against a clinical baseline model
Figure 6   every stable panel genus against every clinical index, and against the
           ether-lipid ratio — the column the prior cohort paper could not carry
Figure S7  relative abundance of the same genera across CKM stages: the single-taxon
           comparison, which is the analysis that does not survive correction here
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
UP = '/mnt/user-data/uploads/CKM+microbiome'; ST = '20260911_v56'
N = pd.read_csv('numbers_20260911_v56.csv').set_index('key').value
nv = lambda k: float(N[k]); ni = lambda k: int(float(N[k]))
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

CURVE = pd.read_csv(f'fig5b_feature_curve_20260911_v54.csv')
DISC = pd.read_csv(f'Table3_discrimination_20260911_v54.csv')
ROCZ = np.load(f'fig5a_roc_20260911_v54.npz')
ENDS = ['Diabetes', 'CKM stage 3 or 4', 'Elevated triglycerides']
MODS = ['Age, sex, BMI', '+ microbiota panel', '+ ether-lipid ratio',
        '+ microbiota panel and ether-lipid ratio']
MCOL = {MODS[0]: GRAY, MODS[1]: GREEN, MODS[2]: BLUE, MODS[3]: ORANGE}
MSHORT = {MODS[0]: 'Age, sex, BMI', MODS[1]: '+ microbiota panel',
          MODS[2]: '+ ether-lipid ratio', MODS[3]: '+ both'}
n743 = ni('n_microbiome'); n825 = ni('n_microbiome_all')

# ============================== FIGURE 5 ==============================
fig = plt.figure(figsize=(7.2, 5.9))
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.36, left=0.10, right=0.98,
                      top=0.93, bottom=0.09)

a = fig.add_subplot(gs[0, 0]); pan(a, 'A')
for en, col in zip(ENDS, [PURPLE, GREEN, ORANGE]):
    d = CURVE[CURVE.endpoint == en].sort_values('k')
    a.plot(d.k, d.auc, 'o-', ms=3.4, lw=1.2, color=col, label=en)
a.set_xlabel('Number of genera in the panel'); a.set_ylabel('Area under the curve')
a.set_ylim(0.55, 0.80)
a.legend(frameon=False, fontsize=6.0, loc='center right', handlelength=1.4)
a.set_title(f'Performance does not depend on panel size (n = {n825})',
            fontsize=7.6, color=INK, pad=4)
a.text(0.03, 0.06, 'the signal is distributed across the community,\nnot carried by a few taxa',
       transform=a.transAxes, fontsize=6.0, color=MUTED, va='bottom', style='italic')
a.set_xlim(0, 65)
clean(a)

for j, en in enumerate(['CKM stage 3 or 4', 'Elevated triglycerides']):
    ax = fig.add_subplot(gs[0, 1] if j == 0 else gs[1, 0])
    pan(ax, 'BC'[j])
    for m in MODS:
        key = f'{en}|{m}'
        if f'{key}|fpr' not in ROCZ: continue
        ax.plot(ROCZ[f'{key}|fpr'], ROCZ[f'{key}|tpr'], lw=1.3, color=MCOL[m],
                label=f'{MSHORT[m]}  '
                      f'{DISC[(DISC["set"]=="MB743")&(DISC.endpoint==en)&(DISC.model==m)].auc.iloc[0]:.3f}')
    ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color=GRID)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
    ev = int(DISC[(DISC['set'] == 'MB743') & (DISC.endpoint == en)].events.iloc[0])
    ax.set_title(f'{en}  ({ev} of {n743})', fontsize=7.6, color=INK, pad=4)
    ax.legend(frameon=False, fontsize=5.8, loc='lower right', handlelength=1.4,
              title='                      AUC', title_fontsize=5.8)
    clean(ax)

d = fig.add_subplot(gs[1, 1]); pan(d, 'D')
D = DISC[DISC['set'] == 'MB743']
ypos, ylab, cols = [], [], []
y = 0
for en in ENDS[::-1]:
    for m in MODS[::-1]:
        r = D[(D.endpoint == en) & (D.model == m)]
        if not len(r): continue
        ypos.append((y, float(r.auc.iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0])))
        ylab.append(MSHORT[m]); cols.append(MCOL[m]); y += 1
    ylab.append(''); cols.append('none'); ypos.append((y, np.nan, np.nan, np.nan)); y += 1
for (yy, au, lo, hi), c in zip(ypos, cols):
    if not np.isfinite(au): continue
    d.plot([lo, hi], [yy, yy], lw=1.1, color=c)
    d.plot(au, yy, 'o', ms=4.2, color=c)
    d.text(hi + 0.006, yy, f'{au:.3f}', va='center', fontsize=5.8, color=INK)
d.set_yticks([p[0] for p in ypos]); d.set_yticklabels(ylab, fontsize=6.0)
d.set_xlim(0.55, 0.83); d.set_xlabel('Area under the curve (95% CI)')
d.set_ylim(-0.8, y - 0.4)
for en, yy in zip(ENDS[::-1], [1.5, 6.5, 11.5]):
    d.text(0.555, yy + 2.0, en, fontsize=6.4, color=INK, fontweight='bold', va='center')
d.set_title(f'Cross-validated discrimination (n = {n743})', fontsize=7.6, color=INK, pad=4)
clean(d); d.spines['left'].set_visible(False); d.tick_params(axis='y', length=0)
plt.savefig(f'Figure5_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()
print('Figure 5 done')

# ============================== FIGURE 6 ==============================
HM = pd.read_csv(f'TableS12_genus_clinical_20260911_v54.csv')
BLOCKS = ['Anthropometric', 'Metabolic', 'Renal', 'Ether lipid']
cols_order, block_of = [], {}
for b in BLOCKS:
    for c in HM[HM.block == b]['index'].drop_duplicates():
        cols_order.append(c); block_of[c] = b
M = HM.pivot(index='genus', columns='index', values='rho').reindex(columns=cols_order)
Pv = HM.pivot(index='genus', columns='index', values='p').reindex(columns=cols_order)
Qv = HM.pivot(index='genus', columns='index', values='q').reindex(columns=cols_order)
order = M['Ether-lipid ratio'].sort_values(ascending=False).index
M, Pv, Qv = M.loc[order], Pv.loc[order], Qv.loc[order]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
fig.subplots_adjust(left=0.27, right=0.90, top=0.80, bottom=0.16)
vmax = float(np.nanmax(np.abs(M.values)))
im = ax.imshow(M.values, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        p = Pv.values[i, j]
        if not np.isfinite(p): continue
        s = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        if s:
            ax.text(j, i, s, ha='center', va='center', fontsize=6.0,
                    color='white' if abs(M.values[i, j]) > 0.6 * vmax else INK,
                    fontweight='bold' if Qv.values[i, j] < 0.05 else 'normal')
ax.set_xticks(range(len(cols_order)))
ax.set_xticklabels(cols_order, rotation=55, ha='right', fontsize=6.2)
ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=6.2, style='italic')
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
start = 0
for b in BLOCKS:
    k = sum(1 for c in cols_order if block_of[c] == b)
    ax.add_patch(Rectangle((start - 0.5, -1.35), k, 0.75, clip_on=False,
                           fc={'Anthropometric': '#4B5563', 'Metabolic': '#374151',
                               'Renal': '#4B5563', 'Ether lipid': GREEN}[b], ec='none'))
    ax.text(start - 0.5 + k / 2, -0.98, 'Ether' if b == 'Ether lipid' else b,
            ha='center', va='center', fontsize=6.4 if k > 1 else 5.6,
            color='white', fontweight='bold', clip_on=False)
    if start: ax.axvline(start - 0.5, color='white', lw=1.6)
    start += k
cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
cb.set_label('Spearman ρ', fontsize=6.4); cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
ax.set_title(f'Stable panel genera against clinical indices and the ether-lipid ratio\n'
             f'n = {n825} (ether-lipid column n = {n743}); '
             f'* $P$ < 0.05, ** $P$ < 0.01, *** $P$ < 0.001; bold marks $q$ < 0.05',
             fontsize=7.4, color=INK, pad=22)
plt.savefig(f'Figure6_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()
print('Figure 6 done')

# ============================== FIGURE S7 ==============================
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
from stage_v10 import master_stage
A = A.assign(CKM5=master_stage(A, 'X6 MB825'))
from xnames import labels as taxa_labels
GC = pd.read_csv('/mnt/user-data/uploads/microbiome_hub_v1/data/abundance_genus_counts.csv',
                 index_col=0)
GX = GC.loc[A.sample_id]; GX = GX.loc[:, (GX > 0).mean(0) >= 0.10]
REL = (GX.div(GX.sum(1), axis=0) * 100); REL.index = A.subject_id.values
REL.columns = taxa_labels(list(GX.columns))
AB = pd.read_csv(f'TableS13_abundance_stage_20260911_v54.csv')
show = AB[AB.prevalence >= 0.50].sort_values('kruskal_p').head(6)
stage = pd.Series(A.CKM5.values, index=A.subject_id.values)

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
fig.subplots_adjust(hspace=0.62, wspace=0.34, left=0.09, right=0.98, top=0.84, bottom=0.12)
for ax, (_, r) in zip(axes.ravel(), show.iterrows()):
    v = REL[r.genus]
    data = [v[stage == s].dropna().values for s in range(5)]
    bp = ax.boxplot(data, widths=0.62, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.1),
                    whiskerprops=dict(color=GRAY, lw=0.8), capprops=dict(color=GRAY, lw=0.8),
                    boxprops=dict(lw=0.8))
    for patch in bp['boxes']:
        patch.set_facecolor('#E7F5EF'); patch.set_edgecolor(GREEN)
    ax.set_xticklabels([f'{i}\n({len(d)})' for i, d in enumerate(data)], fontsize=6.0)
    ax.set_ylabel('Relative abundance (%)', fontsize=6.4)
    ax.set_title(f'{r.genus}', fontsize=7.0, color=INK, style='italic', pad=3)
    ax.text(0.97, 0.94, f'Kruskal–Wallis $P$ = {r.kruskal_p:.3f}\n$q$ = {r.kruskal_q:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=5.8, color=MUTED,
            bbox=dict(fc='white', ec='none', pad=1.2))
    clean(ax)
for ax in axes.ravel()[len(show):]: ax.axis('off')
fig.suptitle(f'Relative abundance across CKM stage, {ni("n_microbiome_all")} profiled participants '
             f'— no genus survives correction', fontsize=8.2, color=INK, y=0.955)
fig.text(0.5, 0.015, 'x axis: CKM stage (number of participants). The same participants and the '
                     'same taxa carry the panel effects shown in Figures 2 and 5,\nso this is a '
                     'calibrated negative: the community-level readout is informative where the '
                     'single-taxon comparison is not.',
         ha='center', fontsize=5.9, color=MUTED, style='italic', linespacing=1.5)
plt.savefig(f'FigureS7_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()
print('Figure S7 done')
