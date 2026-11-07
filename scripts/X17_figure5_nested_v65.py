# -*- coding: utf-8 -*-
"""Figure 5 與 Table 3 改用 nested CV 重做      20261030_v65

舊版的判別力是兩層分開跑的：panel 分數先用全部 743 人選菌屬、以一組 fold 產生
out-of-fold 分數，再把那個分數當固定欄位、用另一組 fold 跑臨床結局。外審
issue 08 指出外層測試集的人會互相影響彼此的分數，這是對的。

這裡每一個數字都來自 nested_core_v65 的 nested_predictions：外層切分固定之後，
CLR 標準化、依 R1 選菌屬、分數標準化、模型配適全部只在外層訓練集內做。

四個 panel：
  A  面板大小曲線，每一個 k 都跑完整的 nested CV
  B  CKM 第三或第四期的 ROC
  C  三酸甘油酯偏高的 ROC
  D  各模型的 AUC，含整條 pipeline 的 bootstrap 區間，以及 82 人保留集

結論方向與舊版不同：面板在 nested 之下幾乎沒有增益，醚脂比值只在三酸甘油酯
這個結局上有中等幅度的增益。這是該寫進正文的版本。
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nested_core_v65 import (S743, S82, END, MODELS, KGRID, auc, nested_predictions, ST)

plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
MCOL = {'Age, sex, BMI': GRAY, '+ microbiota panel': GREEN,
        '+ ether-lipid ratio': BLUE, '+ microbiota panel and ether-lipid ratio': PURPLE}
NEST = pd.read_csv(f'TableSx_nested_cv_{ST}.csv')
HOLD = pd.read_csv(f'TableSy_holdout82_{ST}.csv')


def roc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]
    o = np.argsort(-s); y = y[o]
    return np.r_[0, np.cumsum(1 - y) / max((1 - y).sum(), 1), 1], \
           np.r_[0, np.cumsum(y) / max(y.sum(), 1), 1]


# ---------------------------------------------- A：nested 的面板大小曲線
curve = []
for k in KGRID:
    for en, y in END.items():
        preds, yv, _ = nested_predictions(S743, y, 'topk_fixed', k=k)
        curve.append(dict(k=k, endpoint=en, auc=auc(yv, preds['+ microbiota panel'])))
CURVE = pd.DataFrame(curve)
CURVE.to_csv(f'fig5a_nested_feature_curve_{ST}.csv', index=False)

# ---------------------------------------------- B、C：nested 的 ROC
ROC = {}
for en in END:
    preds, yv, _ = nested_predictions(S743, END[en], 'p01')
    for mn in MODELS:
        ROC[(en, mn)] = roc(yv, preds[mn])

fig = plt.figure(figsize=(7.2, 6.1))
gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.30,
                      left=0.095, right=0.975, top=0.94, bottom=0.14)


def pan(ax, l, x=-0.16, y=1.045):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


a = fig.add_subplot(gs[0, 0]); pan(a, 'A')
for en, c in zip(END, [ORANGE, BLUE, GREEN]):
    d = CURVE[CURVE.endpoint == en]
    a.plot(d.k, d.auc, 'o-', ms=3.5, lw=1.3, color=c, label=en)
a.set_xlabel('Genera in the panel (top $k$ of each outer training fold)')
a.set_ylabel('Area under the curve')
a.set_ylim(0.58, 0.76)
a.legend(frameon=False, fontsize=6.2, loc='upper left', labelcolor=MUTED)
a.set_title(f'Flat across panel size under nested CV (n = {len(S743)})',
            fontsize=7.6, color=INK, pad=4)
clean(a)

for ax_i, en in [(gs[0, 1], 'CKM stage 3 or 4'), (gs[1, 0], 'Elevated triglycerides')]:
    ax = fig.add_subplot(ax_i); pan(ax, 'B' if en.startswith('CKM') else 'C')
    for mn in MODELS:
        fp, tp = ROC[(en, mn)]
        d = NEST[(NEST.set == 'MB743') & (NEST.panel_rule == 'p01') &
                 (NEST.endpoint == en) & (NEST.model == mn)]
        ax.plot(fp, tp, lw=1.3, color=MCOL[mn],
                label=f'{mn} — {float(d.auc.iloc[0]):.3f}')
    ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color=GRID)
    ev = int(NEST[(NEST.set == 'MB743') & (NEST.endpoint == en)].events.iloc[0])
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title(f'{en} ({ev} events of {len(S743)})', fontsize=7.6, color=INK, pad=4)
    ax.legend(frameon=False, fontsize=5.9, loc='lower right', labelcolor=MUTED,
              handlelength=1.4)
    clean(ax)

d = fig.add_subplot(gs[1, 1]); pan(d, 'D')
# 20261030_v65：原本畫四個模型各自的 AUC 與區間，用「區間重疊」去讀有沒有增益。
# 那是錯的：兩個模型評的是同一批人，AUC 高度相關，分開看會嚴重低估差值的精確度。
# 改畫配對的 ΔAUC——每一次 bootstrap 重抽裡跑完整 nested CV，再把兩個模型相減。
DA = pd.read_csv(f'TableSw_delta_auc_{ST}.csv')
rows = []
for en in END:
    for mn in MODELS:
        if mn == 'Age, sex, BMI': continue
        r = DA[(DA.endpoint == en) & (DA.model == mn)]
        if len(r): rows.append((en, mn, float(r.delta.iloc[0]),
                                float(r.lo.iloc[0]), float(r.hi.iloc[0])))
y = np.arange(len(rows))[::-1]
for yy, (en, mn, dl, lo, hi) in zip(y, rows):
    solid = lo > 0
    d.plot([lo, hi], [yy, yy], lw=1.1, color=MCOL[mn], zorder=2)
    d.plot([dl], [yy], 'o', ms=4.0, color=MCOL[mn] if solid else 'white',
           mec=MCOL[mn], mew=1.3, zorder=3)
d.axvline(0, color=GRAY, lw=0.9, ls='--')
h = [plt.Line2D([], [], color=MCOL[m], lw=1.6, label=m) for m in MODELS
     if m != 'Age, sex, BMI']
d.set_yticks([7.0, 4.0, 1.0])
d.set_yticklabels([e.replace(' ', '\n', 1) for e in END], fontsize=6.4)
d.set_ylim(-0.8, 8.8)
d.set_xlim(-0.022, 0.082)
d.set_xlabel('Gain in area under the curve over age, sex and BMI\n(paired bootstrap, 95% CI)')
d.set_title('The host ratio adds discrimination for triglycerides;\nthe panel alone does not',
            fontsize=7.6, color=INK, pad=4)
d.legend(handles=h, frameon=False, fontsize=5.8, labelcolor=MUTED, handlelength=1.4,
         loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=1, borderpad=0.1)
ho = HOLD[HOLD.panel_rule == 'p01']
d.text(0.0, -0.62, 'Filled markers: the 95% interval excludes zero.  Independent 82 without '
                   'metabolomics,\nnever used in panel selection — ' +
       ';  '.join(f'{e}: {float(ho[(ho.endpoint == e) & (ho.model == "Age, sex, BMI")].auc.iloc[0]):.3f}'
                  f' → {float(ho[(ho.endpoint == e) & (ho.model == "+ microbiota panel")].auc.iloc[0]):.3f}'
                  for e in END),
       transform=d.transAxes, ha='left', va='top', fontsize=5.5, color=MUTED,
       linespacing=1.7)
clean(d)

plt.savefig(f'Figure5_nested_{ST}.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('chk/F5_v65.png', dpi=190, bbox_inches='tight', pad_inches=0.10)
plt.close()

T3 = NEST[(NEST.panel_rule == 'p01')].drop(columns=['panel_rule', 'k_per_fold'])
T3.to_csv(f'Table3_discrimination_nested_{ST}.csv', index=False)
print(f'寫出 Figure5_nested_{ST}.svg、Table3_discrimination_nested_{ST}.csv、'
      f'fig5a_nested_feature_curve_{ST}.csv')
print(CURVE.pivot(index='k', columns='endpoint', values='auc').round(3).to_string())
