# -*- coding: utf-8 -*-
"""Figure 2 — 訊號是真的、是特異的、不是造出來的      20261030_v65

  A  八個比值各自在 P < 0.01 的菌屬命中數（原 2A，不動）
  B  兩種菌相分數對八個比值的森林圖（新）：菌相只碰 R1
  C  穩健度階梯（原 2B），改以事前三菌屬分數為主、CV 面板為次
  D  置換負控制（新）：把 R1 打亂後用同一條流程重建分數，虛無分布對觀察值

原 2C（37 個分子的森林圖）與 2D（絕對濃度分布）移到補充圖 S12。
所有數字讀自結果檔。
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
ST = '20261030_v65'
N = pd.read_csv(f'numbers_v65_all_{ST}.csv').set_index('key').value
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
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)


SP = pd.read_csv('out/TableS2_specificity_20261030_v65.csv')
ROB = pd.read_csv('out/TableS6_robustness_20261030_v65.csv')
M8 = pd.read_csv(f'TableSm_panel_to_eight_ratios_{ST}.csv')
PERM = np.load(f'perm_null_{ST}.npz')
LAB = SP.drop_duplicates('ratio').set_index('ratio').label

fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.55, left=0.13, right=0.98,
                      top=0.94, bottom=0.09)

# ---------------------------------------------------------------- A
a = fig.add_subplot(gs[0, 0]); pan(a, 'A')
cnt = SP[SP.p < 0.01].groupby('label').size().reindex(LAB.values).fillna(0).sort_values()
a.barh(np.arange(len(cnt)), cnt.values, 0.62,
       color=[GREEN if l.startswith('R1') else GRAY for l in cnt.index])
a.axvline(nv('n_specificity_expected'), color=ORANGE, ls='--', lw=1.0)
a.text(nv('n_specificity_expected') + 0.6, 2.6, 'expected\nby chance', fontsize=6,
       color=ORANGE, va='center')
a.set_yticks(np.arange(len(cnt))); a.set_yticklabels(cnt.index, fontsize=6.2)
a.set_xlabel(f'Genera with $P$ < 0.01  (of {ni("n_genera_tested")} tested)')
a.set_title(f'{ni("spec_R1_count")} of {ni("n_specificity_total")} genus hits fall on one ratio',
            fontsize=8, color=INK, pad=4)
a.text(0.97, 0.06, f'{ni("n_specificity_q05")} of {ni("n_tests_specificity"):,} tests at $q$ < 0.05',
       transform=a.transAxes, ha='right', fontsize=6.4, color=INK)
clean(a)

# ---------------------------------------------------------------- B
b = fig.add_subplot(gs[0, 1]); pan(b, 'B')
order = ['R1_PCae_lysoPC_z', 'R2_PCae362_361_z', 'R3_SM_over_PC_z', 'R4_SM240_160_z',
         'R5_lysoPC204_PCaa384_z', 'R6_PCaa384_383_z', 'R7_C16_C0_z', 'R8_C2_C0_z']
y = np.arange(len(order))[::-1]
for ex, col, dy, mk in [('Pre-specified score', GREEN, +0.17, 'o'), ('CV panel', GRAY, -0.17, 's')]:
    d = M8[M8.exposure == ex].set_index('ratio').loc[order]
    sig = (d.q_across_8 < 0.05).values
    b.errorbar(d.beta, y + dy, xerr=1.96 * d.se, fmt='none', lw=1.0, ecolor=col, capsize=1.8, zorder=2)
    b.scatter(d.beta, y + dy, s=22, marker=mk, c=[col if s_ else 'white' for s_ in sig],
              edgecolors=col, linewidths=1.0, zorder=3, label=ex)
b.axvline(0, color=GRAY, lw=0.8, ls='--')
b.set_yticks(y); b.set_yticklabels([LAB[r] for r in order], fontsize=6.2)
b.set_xlabel('Effect of the microbiota score on each ratio (SD per SD)')
b.set_title(f'The microbiota reaches one ratio (n = {n743})', fontsize=8, color=INK, pad=4)
b.set_ylim(-1.9, len(order) - 0.4)
b.legend(frameon=False, fontsize=6.0, loc='lower right', labelcolor=MUTED, handletextpad=0.3,
         borderaxespad=0.2)
b.text(0.02, 0.03, 'filled: $q$ < 0.05\nacross the eight', transform=b.transAxes,
       fontsize=5.8, color=MUTED, va='bottom')
b.set_xlim(-0.14, 0.30)
clean(b)

# ---------------------------------------------------------------- C
c = fig.add_subplot(gs[1, 0]); pan(c, 'C')
models = ['Age, sex, batch, BMI', '+ 8 metabolic drugs', '+ all 28 medications',
          '+ Firmicutes/Bacteroidota ratio', '+ 10 genetic principal components', '+ log(triglycerides)', 'no BMI adjustment']
short = ['Base model (age, sex, batch, BMI)', '+ 8 metabolic drugs', '+ all 28 medications',
         '+ Firmicutes/Bacteroidota', '+ 10 genetic ancestry PCs', '+ log(triglycerides)', 'no BMI adjustment']
y = np.arange(len(models))[::-1]
for ex, col, dy, mk in [('Pre-specified score', GREEN, +0.17, 'o'), ('CV panel', GRAY, -0.17, 's')]:
    sel = ROB[(ROB.Exposure == ex) & (ROB.Outcome == 'PC ae/lysoPC')].set_index('Model').loc[models]
    c.errorbar(sel.Beta, y + dy, xerr=1.96 * sel.SE, fmt='none', lw=1.0, ecolor=col, capsize=1.8, zorder=2)
    c.scatter(sel.Beta, y + dy, s=22, marker=mk, color=col, zorder=3, label=ex)
    if ex == 'Pre-specified score':
        for yy, (_, r) in zip(y, sel.iterrows()):
            txt = f'$P$ = {sci(r.P)}' + (f'  (n = {int(r.N)})' if int(r.N) != n743 else '')
            c.text(0.33, yy + dy, txt, va='center', fontsize=5.6, color=INK)
c.axvline(0, color=GRAY, lw=0.8, ls='--')
c.set_yticks(y); c.set_yticklabels(short, fontsize=6.2)
c.set_xlim(-0.02, 0.46); c.set_xlabel('Effect on ether-lipid ratio (SD per SD)')
c.set_title('Robust to medication, composition, ancestry and adiposity', fontsize=8, color=INK, pad=4)
c.set_ylim(-1.6, len(models) - 0.4)
c.legend(frameon=False, fontsize=6.0, loc='lower left', labelcolor=MUTED, handletextpad=0.3,
         borderaxespad=0.2)
clean(c)

# ---------------------------------------------------------------- D
d = fig.add_subplot(gs[1, 1]); pan(d, 'D')
null = PERM['null_abs_beta']; obs = float(PERM['observed_beta'])
d.hist(null, bins=24, color=GRID, edgecolor='white', lw=0.5)
p95 = float(np.percentile(null, 95))
d.axvline(p95, color=ORANGE, ls='--', lw=1.0)
d.axvline(abs(obs), color=GREEN, lw=1.6)
d.axvline(abs(nv('rob_pre_base_beta')), color=GREEN, lw=1.6, ls=':')
ymax = d.get_ylim()[1]
d.text(p95 + 0.003, ymax * 0.92, f'95th percentile\nof the null\n{p95:.3f}', fontsize=5.8,
       color=ORANGE, va='top')
d.text(abs(obs) + 0.003, ymax * 0.55, f'CV panel\nobserved {abs(obs):.3f}\npermutation $P$ = '
       f'{nv("perm_permutation_p"):.3f}', fontsize=5.8, color=GREEN, va='top')
d.text(abs(nv('rob_pre_base_beta')) + 0.003, ymax * 0.22,
       f'pre-specified\nscore {abs(nv("rob_pre_base_beta")):.3f}\n(never selected)',
       fontsize=5.8, color=GREEN, va='top')
d.set_xlabel('|β| of a panel rebuilt on permuted R1, tested against the true R1')
d.set_ylabel(f'Permutations (of {len(null)})')
d.set_title('What the selection procedure alone can manufacture', fontsize=8, color=INK, pad=4)
d.set_xlim(0, max(0.2, abs(nv('rob_pre_base_beta')) + 0.06))
clean(d)

plt.savefig(f'Figure2_{ST}.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('chk/F2_v65.png', dpi=180, bbox_inches='tight', pad_inches=0.10)
plt.close()
print(f'Figure2_{ST}.svg')
