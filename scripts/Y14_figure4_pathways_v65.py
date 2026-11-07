# -*- coding: utf-8 -*-
"""Figure 4 — 事前三菌屬分數的 MetaCyc 路徑富集，與三個菌屬各自扛哪些路徑          20261031_v65

  A  對事前分數在主要模型與 NSTI 校正模型都 q < 0.05 的路徑，依 MetaCyc 名稱關鍵字分類排列，
     長條 = −log10 q*（NSTI 校正），顏色 = 方向（綠 正、紫 負）
  B  同一組路徑 × 三個事前菌屬（各自的 CLR 標準化豐度）的 β 熱圖；* = 該菌屬兩個模型都 q < 0.05
讀 TableSmc_metacyc_taxa（X28）。
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
ST = '20261031_v65'
M = pd.read_csv(f'TableSmc_metacyc_taxa_{ST}.csv')
P = M[(M.exposure == 'Pre-specified score') & M.survives].copy()
GEN = [('Lachnospiraceae NK4A136 group', 'NK4A136'), ('Christensenellaceae R-7 group', 'Christensenellaceae R-7'), ('Enterocloster', 'Enterocloster')]
CATS = ['Carbohydrate metabolism and fermentation', 'Amino acid metabolism', 'Lipid, fatty acid and isoprenoid metabolism',
        'Nucleotide metabolism', 'Cofactor and vitamin biosynthesis', 'Cell envelope biosynthesis', 'Energy metabolism', 'Other']
CATS = [c for c in CATS if c in set(P.category)]
P['cat_i'] = P.category.map({c: i for i, c in enumerate(CATS)})
P = P.sort_values(['cat_i', 'beta'], ascending=[True, False]).reset_index(drop=True)
P['label'] = P.name.str.replace('&beta;', 'β').str.replace('&alpha;', 'α')
P['label'] = P.label.str.replace('superpathway of ', 'superpathway: ').where(P.label.str.len() <= 44, P.label.str[:42] + '…')
n = len(P); n825 = int(P.n.iloc[0])

fig = plt.figure(figsize=(7.4, 0.145 * n + 1.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.40], wspace=0.06, left=0.42, right=0.96, top=0.94, bottom=0.08)
a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1], sharey=a)
y = np.arange(n)[::-1]
val = -np.log10(P.q_nsti_adj.values)
a.barh(y, val, 0.70, color=[GREEN if v > 0 else PURPLE for v in P.beta], edgecolor='white', linewidth=0.4)
for yy, (_, r) in zip(y, P.iterrows()):
    a.text(-np.log10(r.q_nsti_adj) + 0.15, yy, f'{r.beta:+.2f}'.replace('-', '−'), va='center', fontsize=4.8, color=MUTED)
a.set_yticks(y); a.set_yticklabels(P.label, fontsize=5.4)
a.set_xlabel('−log$_{10}$ $q$, coverage-adjusted model')
a.axvline(-np.log10(0.05), color=GRAY, lw=0.7, ls='--')
# category bands and labels
start = 0
for c in CATS:
    k = int((P.category == c).sum()); lo, hi = n - start - k, n - start - 1
    a.axhspan(lo - 0.5, hi + 0.5, color='#F3F4F6' if CATS.index(c) % 2 else 'white', zorder=0)
    a.text(-0.015, (lo + hi) / 2, c.replace(' and ', ' &\n').replace('metabolism', 'metab.'), transform=a.get_yaxis_transform(), ha='right', va='center',
           fontsize=5.2, color=INK, fontweight='bold')
    start += k
# move tick labels right of the band label: place y tick labels inside the band via pad
a.tick_params(axis='y', pad=92, length=0)
a.set_ylim(-0.7, n - 0.3); a.set_xlim(0, max(val) * 1.18)
a.set_title(f'MetaCyc pathways associated with the pre-specified score\n({n} of 365 survive coverage adjustment; n = {n825})', fontsize=7.6, color=INK, pad=5, loc='left', x=-0.55)
a.spines[['top', 'right']].set_visible(False); a.tick_params(axis='x', length=2.5, colors=MUTED)
hpos = plt.Rectangle((0, 0), 1, 1, color=GREEN); hneg = plt.Rectangle((0, 0), 1, 1, color=PURPLE)
a.legend([hpos, hneg], ['rises with the score', 'falls with the score'], frameon=False, fontsize=5.8, loc='lower right', labelcolor=MUTED)
# ---- B heat-map
H = np.column_stack([M[M.exposure == g].set_index('pathway').loc[P.pathway, 'beta'].values for g, _ in GEN])
SV = np.column_stack([M[M.exposure == g].set_index('pathway').loc[P.pathway, 'survives'].values for g, _ in GEN])
vmax = np.abs(H).max()
b.imshow(H[::-1][::-1] if False else H, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto', extent=(-0.5, 2.5, -0.5, n - 0.5), origin='upper')
# imshow with origin upper puts row 0 at top -> row i at y = n-1-i which matches y order
for i in range(n):
    for j in range(3):
        if SV[i, j]: b.text(j, n - 1 - i, '*', ha='center', va='center', fontsize=7, color=INK, fontweight='bold')
b.set_xticks(range(3)); b.set_xticklabels([f'$\\it{{{s.split()[0]}}}$' if s == 'Enterocloster' else s for _, s in GEN], rotation=45, ha='right', fontsize=5.6)
b.tick_params(axis='y', labelleft=False, length=0); b.tick_params(axis='x', length=2, colors=MUTED)
b.set_title('Per-genus β\n(* survives)', fontsize=6.8, color=INK, pad=5)
for sp in b.spines.values(): sp.set_visible(False)
cb = fig.colorbar(b.images[0], ax=b, fraction=0.08, pad=0.04, shrink=0.25, anchor=(0, 1.0))
cb.ax.tick_params(labelsize=5, length=1.5); cb.set_label('β (SD per SD)', fontsize=5.4)
fig.text(0.005, 0.985, 'A', fontsize=10, fontweight='bold', va='top', color=INK)
fig.text(0.765, 0.985, 'B', fontsize=10, fontweight='bold', va='top', color=INK)
plt.savefig(f'Figure4_pathways_{ST}.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('chk/F4path_v65.png', dpi=170, bbox_inches='tight', pad_inches=0.10); plt.close()
print(f'Figure4_pathways_{ST}.svg  ({n} pathways)')
