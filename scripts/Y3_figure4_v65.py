# -*- coding: utf-8 -*-
"""Figure 4 — 醚脂比值標記 CKM 的哪一段（新的核心圖）      20261030_v65

  A  R1 對 CKM 五級分期與第 2 期的三個判準（三酸甘油酯、糖尿病、HbA1c）。
     後三者標明是判準的拆解，不是並列結局。
  B  八條平行中介：四個 CKM 結局的總間接效果與 c′；第 2 期那一刀拆到各通路。
  C  八個比值在四個分期切點上的效果（代謝體 ∩ 臨床），顯示 R1 撐 ≥2、R3 與 R7 撐 ≥3／≥4。
  D  兩條代謝體軸：從疾病端建軸之後，軸本身在 MB743 的驗證、菌相對軸的關聯，以及軸的權重。

所有數字讀自結果檔。
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
ST = '20261030_v65'
N = pd.read_csv(f'numbers_v65_all_{ST}.csv').set_index('key').value
nv = lambda k: float(N[k]); ni = lambda k: int(float(N[k]))
SHORT = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
RLAB = {'R1': 'R1 PC ae / lysoPC', 'R2': 'R2 PC ae C36:2/C36:1', 'R3': 'R3 SM / PC',
        'R4': 'R4 SM C24:0/C16:0', 'R5': 'R5 lysoPC C20:4 / PC aa C38:4',
        'R6': 'R6 PC aa C38:4/C38:3', 'R7': 'R7 C16/C0', 'R8': 'R8 C2/C0'}


def sci(p, d=1):
    if p >= 0.01: return f'{p:.3g}'
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f'{m:.{d}f} × 10$^{{{e}}}$'


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


def pan(ax, l, x=-0.18, y=1.05):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)


S8 = pd.read_csv('out/TableS8_ratio_outcome_20261030_v65.csv')
PM = pd.read_csv(f'TableSh_parallel_mediation_{ST}.csv'); PM = PM[PM.exposure == 'Pre-specified score']
SO = pd.read_csv(f'TableSo_eight_ratio_per_ratio_{ST}.csv'); SO = SO[SO['sample'].str.startswith('Metabolome')]
SL = pd.read_csv(f'TableSl_metabolite_disease_axis_{ST}.csv')
SK = pd.read_csv(f'TableSk_axis_weights_{ST}.csv')

fig = plt.figure(figsize=(9.0, 7.6))
gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.70, left=0.15, right=0.985,
                      top=0.945, bottom=0.075)

# ---------------------------------------------------------------- A
gsa = gs[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.95)
a = fig.add_subplot(gsa[0, 0]); pan(a, 'A', x=-0.75)
r1 = S8[S8.ratio == 'R1'].set_index('outcome')
rows = [('CKM stage (5-level)', 'CKM stage (0–4)', BLUE),
        ('log(triglycerides)', 'Triglycerides', GREEN), ('Diabetes', 'Diabetes', GREEN),
        ('HbA1c', 'HbA1c', GREEN)]
y = np.arange(len(rows))[::-1]
for yy, (k, lab, col) in zip(y, rows):
    r = r1.loc[k]
    a.errorbar(r.beta, yy, xerr=1.96 * r.se, fmt='o', ms=4.6, lw=1.1, color=col, ecolor=col, capsize=2)
    a.text(0.012, yy, f'$q$ = {sci(r.q)}', fontsize=5.4, color=INK, va='center', ha='left')
a.axvline(0, color=GRAY, lw=0.8, ls='--')
a.set_yticks(y); a.set_yticklabels([lab for _, lab, _ in rows], fontsize=6.6)
a.axhspan(-0.5, 2.5, color=GREEN, alpha=0.06, lw=0)
a.text(-0.39, 2.42, 'stage-2 criteria', fontsize=5.8, color=GREEN, ha='left', va='top')
a.set_xlim(-0.40, 0.17); a.set_xlabel(f'Effect of R1 (SD per SD;\ndiabetes log-odds); n = {int(r1.n.min()):,}–{int(r1.n.max()):,}')
a.set_title('R1 tracks CKM stage\nthrough its stage-2 criteria', fontsize=7.2, color=INK, pad=4, loc='left', x=-0.3)
clean(a)
# A 右半：R1 與事前分數對四個切點（TableSq）
SQ = pd.read_csv(f'TableSq_stage_boundaries_{ST}.csv')
ab = fig.add_subplot(gsa[0, 1]); pan(ab, 'B', x=-0.40)
cuts = ['stage >= 1', 'stage >= 2', 'stage >= 3', 'stage >= 4']
xb = np.arange(4)
for ex, col, dx, mk, lab_ in [('Ether-lipid ratio', BLUE, -0.17, 'o', 'R1'), ('Pre-specified score', GREEN, +0.17, 's', 'microbiota score')]:
    d_ = SQ[SQ.exposure == ex].set_index('boundary').loc[cuts]
    ab.errorbar(xb + dx, d_.beta, yerr=1.96 * d_.se, fmt='none', ecolor=col, lw=1.0, capsize=1.8, zorder=2)
    ab.scatter(xb + dx, d_.beta, s=20, marker=mk, c=[col if p < 0.05 else 'white' for p in d_.p], edgecolors=col, linewidths=1.0, zorder=3, label=lab_)
ab.axhline(0, color=GRAY, lw=0.8, ls='--')
ab.set_xticks(xb); ab.set_xticklabels(['≥ 1', '≥ 2', '≥ 3', '≥ 4'], fontsize=6.6); ab.set_xlabel('CKM stage boundary')
ab.set_ylabel('log-odds per SD (95% CI)')
r1b = SQ[(SQ.exposure == 'Ether-lipid ratio') & (SQ.boundary == 'stage >= 2')].iloc[0]
ab.text(1, r1b.beta - 1.96 * r1b.se - 0.12, f'$P$ = {sci(r1b.p)}', ha='center', va='top', fontsize=5.8, color=BLUE)
# 20261107_v66：切點之間直接比較 R1 係數（同一批人的配對 bootstrap，X31）
BC = pd.read_csv('TableSt_boundary_contrast_20261107_v66.csv').set_index('contrast')
def _pb(p, B):
    return f'< {1 / B:.4f}' if p < 1 / B else f'= {sci(p) if p < 0.01 else f"{p:.3f}"}'
_c3 = BC.loc['beta(stage >= 2) - beta(stage >= 3)']; _c4 = BC.loc['beta(stage >= 2) - beta(stage >= 4)']
ab.text(0.03, 0.97, f'≥ 2 vs ≥ 3: $P$ {_pb(_c3.p_boot, _c3.n_boot)}\n≥ 2 vs ≥ 4: $P$ {_pb(_c4.p_boot, _c4.n_boot)}\n(paired bootstrap)',
        transform=ab.transAxes, ha='left', va='top', fontsize=5.4, color=BLUE, linespacing=1.3)
ab.set_title(f'Only stage ≥ 2\nis reached (n = {int(SQ.n.iloc[0])})', fontsize=7.2, color=INK, pad=4)
ab.legend(frameon=False, fontsize=5.8, loc='lower right', labelcolor=MUTED, handletextpad=0.3)
ab.set_ylim(-1.3, 0.6); clean(ab)

# ---------------------------------------------------------------- B
gsb = gs[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.62], wspace=0.55)
b = fig.add_subplot(gsb[0, 0]); pan(b, 'C', x=-0.42, y=1.12)
oc_order = ['CKM stage >= 2', 'CKM stage (5-level)', 'CKM stage >= 3', 'CKM stage >= 4']
oc_lab = {'CKM stage >= 2': 'CKM ≥ 2', 'CKM stage (5-level)': 'CKM 0–4',
          'CKM stage >= 3': 'CKM ≥ 3', 'CKM stage >= 4': 'CKM ≥ 4'}
pm = PM.set_index('outcome').loc[oc_order]
y = np.arange(len(oc_order))[::-1]
for yy, (k, r) in zip(y, pm.iterrows()):
    excl = (r.ind_lo < 0 and r.ind_hi < 0) or (r.ind_lo > 0 and r.ind_hi > 0)
    b.plot([r.ind_lo, r.ind_hi], [yy, yy], color=PURPLE, lw=1.2, zorder=2)
    b.plot([r.total_indirect], [yy], 'o', ms=5, color=PURPLE if excl else 'white',
           mec=PURPLE, mew=1.2, zorder=3)
    b.plot([r.c_prime], [yy - 0.3], 'D', ms=3.6, color=GRAY, zorder=3)
    # 20261107_v66：不印「% mediated」——比例的 bootstrap 區間跨越 −400% 到 +600%，只報間接效果本身
b.axvline(0, color=GRAY, lw=0.8, ls='--')
b.set_yticks(y); b.set_yticklabels([oc_lab[k] for k in oc_order], fontsize=6.6)
b.set_xlabel('Total indirect effect through all eight ratios\n(95% bootstrap CI)')
b.set_title(f'Eight ratios as parallel mediators\n(pre-specified score, n = {ni("n_microbiome")})',
            fontsize=7.6, color=INK, pad=4, loc='left')
h = [plt.Line2D([], [], marker='o', color=PURPLE, lw=0, ms=5, label='total indirect\n(filled: CI excludes 0)'),
     plt.Line2D([], [], marker='D', color=GRAY, lw=0, ms=3.6, label="direct effect c′")]
b.legend(handles=h, frameon=False, fontsize=5.4, labelcolor=MUTED, handletextpad=0.3,
         loc='lower left', borderpad=0.1)
b.set_xlim(-0.30, 0.16); b.set_ylim(-1.6, 3.8)
# 右格：≥2 那一刀拆到各通路
ins = fig.add_subplot(gsb[0, 1])
r2 = pm.loc['CKM stage >= 2']
vals = [r2[f'ind_{s_}'] for s_ in SHORT]
ins.barh(np.arange(8)[::-1], vals, 0.65, color=[GREEN if s_ == 'R1' else PURPLE for s_ in SHORT])
ins.axvline(0, color=GRAY, lw=0.6)
ins.set_yticks(np.arange(8)[::-1]); ins.set_yticklabels(SHORT, fontsize=5.2)
ins.tick_params(length=1.5, labelsize=5.4, colors=MUTED); 
ins.set_xlabel('indirect via\neach ratio\n(CKM ≥ 2)', fontsize=5.6)
ins.spines[['top', 'right']].set_visible(False)
clean(b)

# ---------------------------------------------------------------- C
c = fig.add_subplot(gs[1, 0]); pan(c, 'D', x=-0.52)
cuts = ['stage >= 1', 'stage >= 2', 'stage >= 3', 'stage >= 4']
B = np.full((8, 4), np.nan); Qm = np.full((8, 4), np.nan)
for i, s_ in enumerate(SHORT):
    for j, ct in enumerate(cuts):
        r = SO[(SO.boundary == ct) & (SO.ratio.str.startswith(s_ + ' '))]
        if len(r): B[i, j] = r.beta.iloc[0]; Qm[i, j] = r.q.iloc[0]
vmax = np.nanmax(np.abs(B))
im = c.imshow(B, cmap='RdBu_r', norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
for i in range(8):
    for j in range(4):
        if np.isfinite(Qm[i, j]) and Qm[i, j] < 0.05:
            c.text(j, i, '*', ha='center', va='center', fontsize=9, color=INK, fontweight='bold')
c.set_xticks(range(4)); c.set_xticklabels(['≥ 1', '≥ 2', '≥ 3', '≥ 4'], fontsize=6.6)
c.set_yticks(range(8)); c.set_yticklabels([RLAB[s_] for s_ in SHORT], fontsize=6.0)
c.set_xlabel('CKM stage boundary')
n_big = int(SO.n.iloc[0])
c.set_title(f'Which ratios carry which boundary\n(n = {n_big:,}; * $q$ < 0.05 among the eight)',
            fontsize=8, color=INK, pad=4)
cb = plt.colorbar(im, ax=c, fraction=0.045, pad=0.03); cb.ax.tick_params(labelsize=5.6, length=1.5)
cb.set_label('log-odds per SD', fontsize=6)
c.tick_params(length=2.5, colors=MUTED)
for sp in c.spines.values(): sp.set_visible(False)

# ---------------------------------------------------------------- D
gsd = gs[1, 1].subgridspec(1, 2, width_ratios=[1.0, 0.62], wspace=0.55)
d = fig.add_subplot(gsd[0, 0]); pan(d, 'E', x=-0.42, y=1.12)
cuts2 = ['stage >= 2', 'stage >= 3', 'stage >= 4']
y = np.arange(len(cuts2))[::-1]
for yy, ct in zip(y, cuts2):
    v = SL[(SL.boundary == ct) & SL.test.str.contains('validates')].iloc[0]
    m = SL[(SL.boundary == ct) & SL.test.str.contains('Pre-specified')].iloc[0]
    d.plot([m.beta - 1.96 * m.se, m.beta + 1.96 * m.se], [yy, yy], color=GREEN, lw=1.2, zorder=2)
    d.plot([m.beta], [yy], 'o', ms=5, color=GREEN if m.p < 0.05 else 'white', mec=GREEN, mew=1.2, zorder=3)
    d.text(-0.215, yy + 0.22, ('microbiota → axis: $P$ = ' + (f'{m.p:.3f}' if m.p >= 0.001 else sci(m.p))),
           fontsize=5.8, color=INK, va='bottom', ha='left')
    d.text(-0.215, yy - 0.22, f'axis → boundary: $P$ = {sci(v.p)}', fontsize=5.0, color=MUTED, va='top', ha='left')
d.axvline(0, color=GRAY, lw=0.8, ls='--')
d.set_yticks(y); d.set_yticklabels(['Eight-ratio axis\nfor CKM ≥ 2', 'Eight-ratio axis\nfor CKM ≥ 3', 'Eight-ratio axis\nfor CKM ≥ 4'], fontsize=6.4)
d.set_xlim(-0.22, 0.10); d.set_ylim(-0.7, 2.7); d.set_xlabel('Effect of the pre-specified\nmicrobiota score on the axis\n(SD per SD, 95% CI)')
d.set_title('The microbiota moves only the axis\nbuilt for the stage ≥ 2 boundary', fontsize=7.6, color=INK, pad=4, loc='left')
clean(d)
# 右格：軸的權重
ins = fig.add_subplot(gsd[0, 1])
Wm = np.array([[SK[(SK.boundary == ct) & (SK.ratio.str.startswith(s_ + '_'))].weight.iloc[0]
                for ct in cuts2] for s_ in SHORT])
wmax = np.abs(Wm).max()
ins.imshow(Wm, cmap='RdBu_r', norm=TwoSlopeNorm(0, -wmax, wmax), aspect='auto')
ins.set_xticks(range(3)); ins.set_xticklabels(['≥2', '≥3', '≥4'], fontsize=5.2)
ins.set_yticks(range(8)); ins.set_yticklabels(SHORT, fontsize=5.0)
ins.tick_params(length=1.5, colors=MUTED, labelsize=5.4); ins.set_xlabel('axis\nweights', fontsize=5.6)
for sp in ins.spines.values(): sp.set_visible(False)

plt.savefig('Figure5_chain_20261107_v66.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('chk/F5chain_v65.png', dpi=180, bbox_inches='tight', pad_inches=0.10)
plt.close()
print('Figure5_chain_20261031_v65.svg')
