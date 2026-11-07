# -*- coding: utf-8 -*-
"""補充圖 S9–S12                                             20261030_v65

  S9   原 Figure 6（八比值 × 臨床結局熱圖）原封不動改名
  S10  時間敏感度：事前分數對採便當次 R1、對 >180 天前的 R1、對調整過去 R1 後的當次 R1（TableSu）
  S11  CV 面板的 nested 面板大小曲線（fig5a）與 82 人獨立保留集（TableSy）
  S12  原 Figure 2C/2D：37 個 PC ae 分子的森林圖與絕對濃度分布，改以事前分數為主（TableS5）
所有數字讀自結果檔。
"""
from __future__ import annotations
import shutil
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
ST = '20261030_v65'
N = pd.read_csv(f'numbers_v65_all_{ST}.csv').drop_duplicates('key', keep='last').set_index('key').value
nv = lambda k: float(N[k]); ni = lambda k: int(float(N[k]))
n743 = ni('n_microbiome')


def sci(p, d=1):
    if p >= 0.01: return f'{p:.3g}'
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f'{m:.{d}f} × 10$^{{{e}}}$'


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


def pan(ax, l, x=-0.18, y=1.05):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)


def save(name):
    plt.savefig(f'{name}_{ST}.svg', bbox_inches='tight', pad_inches=0.10)
    plt.savefig(f'chk/{name}_v65.png', dpi=170, bbox_inches='tight', pad_inches=0.10)
    plt.close(); print(f'{name}_{ST}.svg')


# ================================================================ S9
shutil.copy(f'out/Figure6_{ST}.svg', f'FigureS9_{ST}.svg'); print(f'FigureS9_{ST}.svg (= Figure6)')

# ================================================================ S10 時間敏感度
SU = pd.read_csv(f'TableSu_temporal_{ST}.csv')
models = ['R1 at the stool visit (no lag adjustment)', 'R1 measured >180 d earlier',
          'R1 at the stool visit, adjusted for the earlier R1']
short = ['R1 at the stool visit', 'R1 measured > 180 d\nbefore the stool visit', 'R1 at the stool visit,\nadjusted for the earlier R1']
fig = plt.figure(figsize=(4.2, 2.6)); ax = fig.add_subplot(111)
y = np.arange(3)[::-1]
for ex, col, dy, mk in [('Pre-specified score', GREEN, +0.16, 'o'), ('CV panel', GRAY, -0.16, 's')]:
    d = SU[SU.exposure == ex].set_index('model').loc[models]
    ax.errorbar(d.beta, y + dy, xerr=1.96 * d.se, fmt='none', lw=1.0, ecolor=col, capsize=1.8, zorder=2)
    ax.scatter(d.beta, y + dy, s=22, marker=mk, c=[col if p < 0.05 else 'white' for p in d.p],
               edgecolors=col, linewidths=1.0, zorder=3, label=ex)
    if ex == 'Pre-specified score':
        for yy, (_, r) in zip(y, d.iterrows()):
            ax.text(0.30, yy + dy, f'$P$ = {sci(r.p)}', va='center', fontsize=6, color=INK)
ax.axvline(0, color=GRAY, lw=0.8, ls='--')
ax.set_yticks(y); ax.set_yticklabels(short, fontsize=6.4)
ax.set_xlim(-0.06, 0.42); ax.set_ylim(-0.7, 2.7)
ax.set_xlabel('Effect of the microbiota score on R1 (SD per SD)')
ax.set_title(f'One stool sample, two blood draws (n = {int(SU.n.iloc[0])})', fontsize=8, color=INK, pad=4)
ax.legend(frameon=False, fontsize=6, loc='lower right', labelcolor=MUTED, handletextpad=0.3)
ax.text(0.40, 0.03, 'filled: $P$ < 0.05', transform=ax.transAxes, fontsize=5.8, color=MUTED)
clean(ax); save('FigureS10')

# ================================================================ S11 面板大小曲線 + 82 人保留集
CURVE = pd.read_csv(f'out/fig5a_nested_feature_curve_{ST}.csv')
HOLD = pd.read_csv(f'TableSy_holdout82_{ST}.csv'); ho = HOLD[HOLD.panel_rule == 'p01']
ENDS = ['Diabetes', 'CKM stage 3 or 4', 'Elevated triglycerides']
fig = plt.figure(figsize=(7.6, 2.9))
gs = fig.add_gridspec(1, 2, wspace=0.45, left=0.09, right=0.98, top=0.86, bottom=0.2)
a = fig.add_subplot(gs[0, 0]); pan(a, 'A')
for en, c in zip(ENDS, [ORANGE, BLUE, GREEN]):
    d = CURVE[CURVE.endpoint == en]
    a.plot(d.k, d.auc, 'o-', ms=3.5, lw=1.3, color=c, label=en)
a.set_xlabel('Genera in the CV panel (top $k$ of each outer training fold)')
a.set_ylabel('Area under the curve (nested CV)'); a.set_ylim(0.58, 0.78)
a.legend(frameon=False, fontsize=6.2, loc='upper left', labelcolor=MUTED)
a.set_title(f'Flat across panel size (n = {n743})', fontsize=8, color=INK, pad=4); clean(a)
b = fig.add_subplot(gs[0, 1]); pan(b, 'B')
x = np.arange(3)
for i, (mn, col, dx) in enumerate([('Age, sex, BMI', GRAY, -0.18), ('+ microbiota panel', GREEN, +0.18)]):
    v = [float(ho[(ho.endpoint == e) & (ho.model == mn)].auc.iloc[0]) for e in ENDS]
    b.bar(x + dx, v, 0.34, color=col, label=mn)
    for xi, vi in zip(x + dx, v): b.text(xi, vi + 0.008, f'{vi:.3f}', ha='center', fontsize=5.8, color=INK)
b.set_xticks(x); b.set_xticklabels([f'{e.replace(" triglycerides", chr(10) + "triglycerides")}\n({int(ho[ho.endpoint == e].events.iloc[0])} of {int(ho[ho.endpoint == e].n.iloc[0])})' for e in ENDS], fontsize=6.4)
b.set_ylim(0.4, 0.8); b.set_ylabel('Area under the curve')
b.axhline(0.5, color=GRAY, lw=0.8, ls='--')
b.legend(frameon=False, fontsize=6.2, loc='upper left', labelcolor=MUTED)
b.set_title('Independent hold-out without metabolomics', fontsize=8, color=INK, pad=4); clean(b)
save('FigureS11')

# ================================================================ S12 分子層次（原 2C/2D）
SPC = pd.read_csv(f'out/TableS5_species_{ST}.csv')
EX = 'Pre-specified score'
fig = plt.figure(figsize=(7.2, 3.1))
gs = fig.add_gridspec(1, 2, wspace=0.40, left=0.09, right=0.98, top=0.86, bottom=0.2)
c = fig.add_subplot(gs[0, 0]); pan(c, 'A')
d1 = SPC[(SPC.group == 'PC ae / total lysoPC') & (SPC.score == EX)].sort_values('beta')
yy = np.arange(len(d1))
c.errorbar(d1.beta, yy, xerr=1.96 * d1.se, fmt='none', lw=0.8, ecolor=GRID, zorder=1)
c.scatter(d1.beta, yy, s=11, zorder=2, c=[GREEN if q < 0.05 else 'white' for q in d1.q], edgecolors=GREEN, linewidths=0.8)
c.axvline(0, color=GRAY, lw=0.8, ls='--')
c.set_yticks([]); c.set_xlabel('Effect of the pre-specified score on species / total lysoPC (SD per SD)')
c.set_ylabel(f'{len(d1)} ether phosphatidylcholine species')
nq, nsame = int((d1.q < 0.05).sum()), int((np.sign(d1.beta) == np.sign(d1.beta.median())).sum())
c.set_title(f'{nq} of {len(d1)} species individually significant', fontsize=8, color=INK, pad=4)
c.text(0.03, 0.94, f'all {nsame} in the same direction\nn = {int(d1.n.iloc[0])}', transform=c.transAxes, fontsize=6.2, color=INK, va='top')
clean(c)
d = fig.add_subplot(gs[0, 1]); pan(d, 'B')
pc = SPC[(SPC.group == 'PC ae absolute') & (SPC.score == EX)].beta.values
ly = SPC[(SPC.group == 'lysoPC absolute') & (SPC.score == EX)].beta.values
from scipy import stats
mw = stats.mannwhitneyu(pc, ly, alternative='two-sided').pvalue
parts = d.violinplot([pc, ly], positions=[0, 1], widths=0.7, showextrema=False)
for pcol, col in zip(parts['bodies'], [GREEN, PURPLE]):
    pcol.set_facecolor(col); pcol.set_alpha(0.28); pcol.set_edgecolor(col); pcol.set_lw(0.9)
for i, (v, col) in enumerate([(pc, GREEN), (ly, PURPLE)]):
    d.scatter(np.full(len(v), i) + np.random.default_rng(1).normal(0, 0.045, len(v)), v, s=8, color=col, alpha=0.75, zorder=3)
    d.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.4, zorder=4)
d.axhline(0, color=GRAY, lw=0.8, ls='--')
d.set_xticks([0, 1]); d.set_xticklabels([f'PC ae\n({len(pc)} species)', f'lysoPC\n({len(ly)} species)'], fontsize=6.6)
d.set_ylabel('Effect on absolute concentration (SD per SD)')
d.set_title('Ether species rise, lysoPC species do not', fontsize=8, color=INK, pad=4)
d.text(0.97, 0.05, f'Mann–Whitney $P$ = {sci(mw)}\nn = {n743}', transform=d.transAxes, ha='right', fontsize=6.4, color=INK)
clean(d); save('FigureS12')
pd.DataFrame([dict(key='s12_n_species_q05', value=nq, source='TableS5'), dict(key='s12_n_species_same_direction', value=nsame, source='TableS5'),
              dict(key='s12_n_species', value=len(d1), source='TableS5'), dict(key='s12_mw_p', value=mw, source='TableS5'),
              dict(key='s12_n_pcae_abs', value=len(pc), source='TableS5'), dict(key='s12_n_lysopc_abs', value=len(ly), source='TableS5')]
             ).to_csv(f'numbers_s12_{ST}.csv', index=False)
