# -*- coding: utf-8 -*-
"""Figure S4 依 Table S16 的同一個模型重畫      20261030_v65

外審 issue 12。舊的 Figure S4 印的是 P = 0.155 與 0.661，Table S16 是
0.10185 與 0.87225，兩者對不上，而且舊圖沒有任何腳本在產生它——它是從更早
一輪分析沿用下來的圖檔。沿用的圖不能再留著，因為沒人能重現它。

這裡重畫，數字全部來自現在的模型：

  A  醚脂比值對交叉驗證面板分數，依 rs174547 基因型分層，各組畫自己的斜率
  B  兩個事前登錄的交互作用項，附 95% 信賴區間，並標出在這個樣本數下
     80% 檢定力可以偵測到的最小交互作用效果量——審閱人要的 power 就是這個

舊圖的 panel A（rs174547 對去飽和指數，依面板中位數分層）與 panel C
（代謝物 × 菌相對臨床結局的交互作用）在現行管線裡沒有對應的分析可以重現，
所以不畫。這件事要寫進圖說，不能假裝那兩個 panel 還在。
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
UP = '/mnt/user-data/uploads/CKM+microbiome'
SRC = '20260911_v54'
OUT = 'FigureS4_20261030_v65.svg'

rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))

IX = pd.read_csv(f'TableS16_rs174547_interaction_{SRC}.csv')
assert len(IX) == 2, IX
PAN = pd.read_csv(f'panel_scores_{SRC}.csv', index_col=0)
PAN.index = PAN.index.astype(str)
PHE = pd.read_csv(f'pheno_all_{SRC}.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
ph = PHE.set_index('subject_id')
R1 = pd.Series(rint(ph.R1_PCae_lysoPC_z.values), index=ph.index)
G = pd.read_csv(f'{UP}/analysis/pipeline/geno_panel500.csv.gz',
                usecols=['subject_id', 'rs174547'])
G['subject_id'] = G.subject_id.astype(str); G = G.set_index('subject_id').rs174547

sub = [s for s in PAN.index if s in R1.index and s in G.index
       and np.isfinite(PAN.panel_oof.get(s, np.nan)) and np.isfinite(G.get(s, np.nan))]
d = pd.DataFrame({'panel': PAN.panel_oof.reindex(sub).values,
                  'score': PAN.score_prior.reindex(sub).values,
                  'r1': R1.reindex(sub).values,
                  'g': G.reindex(sub).values.round().astype(int)}, index=sub).dropna(subset=['panel', 'r1'])
n_used = len(d)
assert n_used == int(IX.n.iloc[0]), f'樣本數 {n_used} 與 Table S16 的 {IX.n.iloc[0]} 不一致'

fig = plt.figure(figsize=(7.2, 3.1))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.34,
                      left=0.085, right=0.985, top=0.86, bottom=0.17)

# ------------------------------------------------- A
a = fig.add_subplot(gs[0, 0])
a.text(-0.16, 1.06, 'A', transform=a.transAxes, fontsize=10, fontweight='bold',
       va='bottom', ha='left', color=INK)
COLS = {0: BLUE, 1: ORANGE, 2: GREEN}
# 基因檔只給 0/1/2 的計數，沒有給哪一個是 C 哪一個是 T，所以不寫鹼基
LAB = {0: '0 copies', 1: '1 copy', 2: '2 copies'}
xs = np.linspace(d.panel.quantile(0.01), d.panel.quantile(0.99), 2)
for gval, grp in d.groupby('g'):
    if len(grp) < 20: continue
    a.scatter(grp.panel, grp.r1, s=7, color=COLS.get(gval, GRAY), alpha=0.42,
              edgecolors='none', zorder=2)
    b1, b0 = np.polyfit(grp.panel, grp.r1, 1)
    a.plot(xs, b0 + b1 * xs, color=COLS.get(gval, GRAY), lw=1.6, zorder=3,
           label=f'{LAB.get(gval, gval)}: n = {len(grp)}, slope '
           + f'{b1:+.3f}'.replace('-', '−'))
a.axhline(0, color=GRID, lw=0.7, zorder=1)
a.set_xlabel('Cross-validated microbiota panel (SD)')
a.set_ylabel('Ether-lipid ratio (rank-INT SD)')
a.set_title(f'rs174547 does not modify the panel–ratio association\n'
            f'(interaction $P$ = {IX.p.iloc[0]:.3f}; n = {n_used})',
            fontsize=7.6, color=INK, pad=4)
a.legend(frameon=False, fontsize=6.0, loc='upper left', labelcolor=MUTED,
         handletextpad=.5, borderpad=0.1)
a.spines[['top', 'right']].set_visible(False); a.tick_params(length=2.5, colors=MUTED)
a.xaxis.label.set_color(INK); a.yaxis.label.set_color(INK)

# ------------------------------------------------- B
b = fig.add_subplot(gs[0, 1])
b.text(-0.22, 1.06, 'B', transform=b.transAxes, fontsize=10, fontweight='bold',
       va='bottom', ha='left', color=INK)
y = np.arange(len(IX))[::-1]
lo, hi = IX.beta - 1.96 * IX.se, IX.beta + 1.96 * IX.se
b.errorbar(IX.beta, y, xerr=1.96 * IX.se, fmt='o', ms=4.5, lw=1.2,
           color=PURPLE, ecolor=PURPLE, capsize=2.5, zorder=3)
b.axvline(0, color=GRAY, lw=0.9, ls='--', zorder=1)
# 80% 檢定力下可偵測的最小交互作用（雙尾 alpha 0.05），用實際的標準誤算
mde = (stats.norm.isf(0.025) + stats.norm.isf(0.20)) * IX.se
for yy, (_, r_) in zip(y, IX.iterrows()):
    b.plot([-mde[_], mde[_]], [yy - 0.26, yy - 0.26], color=GRID, lw=3.0,
           solid_capstyle='butt', zorder=2)
    b.text(hi[_] + 0.012, yy + 0.08, f'$P$ = {r_.p:.3f}', va='center', ha='left',
           fontsize=6.2, color=INK)
b.set_yticks(y)
b.set_yticklabels(['rs174547 ×\ncross-validated panel',
                   'rs174547 ×\npre-specified score'], fontsize=6.6)
b.set_ylim(-0.75, len(IX) - 0.35)
b.set_xlabel('Interaction effect on the ether-lipid ratio\n(SD per SD per allele, 95% CI)')
b.set_title('Neither pre-registered interaction\nis significant', fontsize=7.6, color=INK, pad=4)
b.text(0.5, 0.02, f'grey bar: smallest interaction detectable with 80% power\n'
                  f'(±{mde.iloc[0]:.3f} and ±{mde.iloc[1]:.3f}); n = {n_used}',
       transform=b.transAxes, ha='center', va='bottom', fontsize=5.9, color=MUTED)
b.spines[['top', 'right']].set_visible(False); b.tick_params(length=2.5, colors=MUTED)
b.xaxis.label.set_color(INK); b.yaxis.label.set_color(INK)

plt.savefig(OUT, bbox_inches='tight', pad_inches=0.10)
plt.savefig('chk/S4_v65.png', dpi=190, bbox_inches='tight', pad_inches=0.10)
plt.close()

F = [dict(key='figS4_n', value=n_used, unit='people', source='Figure S4'),
     dict(key='figS4_mde_panel', value=round(float(mde.iloc[0]), 4), unit='SD per SD',
          source='smallest interaction detectable with 80% power'),
     dict(key='figS4_mde_score', value=round(float(mde.iloc[1]), 4), unit='SD per SD',
          source='smallest interaction detectable with 80% power')]
pd.DataFrame(F).to_csv('numbers_figS4_20261030_v65.csv', index=False)
print(f'寫出 {OUT}')
print(IX.to_string(index=False))
print(f'80% 檢定力可偵測的最小交互作用：面板 ±{mde.iloc[0]:.3f}、事前分數 ±{mde.iloc[1]:.3f}')
print('各基因型人數：', d.g.value_counts().sort_index().to_dict())
