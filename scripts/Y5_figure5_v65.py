# -*- coding: utf-8 -*-
"""Figure 5 — 八個比值接得到哪一層的判別力                20261030_v65

  A  CKM ≥ 2 的 ROC：年齡性別 BMI、+R1、+八比值
  B  CKM 3 或 4 的 ROC
  C  三酸甘油酯偏高的 ROC
  D  四個結局 × 三個模型的配對 ΔAUC（含 CV 面板，TableSw）

ROC 曲線的 out-of-fold 預測值用 X21 同一段程式（同 seed）重算，AUC 與 TableSn 逐一核對。
面板大小曲線與 82 人保留集移到補充圖。所有數字讀自結果檔。
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ST = '20261030_v65'
src = open('X21_eight_ratios_stage_v65.py', encoding='utf-8').read()
head = src[:src.index('\nB = 500')]
head = head.replace("BL.to_csv(", "pass  # BL.to_csv(").replace("PR.to_csv(", "pass  # PR.to_csv(")
g = {'__name__': 'x21_head'}
exec(compile(head, 'X21_head', 'exec'), g)
END, MODELS, cv_auc, auc = g['END'], g['MODELS'], g['cv_auc'], g['auc']
SN = pd.read_csv(f'TableSn_eight_ratio_auc_{ST}.csv')
SW = pd.read_csv(f'TableSw_delta_auc_{ST}.csv')
n743 = len(g['S743'])

plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
MCOL = {'Age, sex, BMI': GRAY, '+ R1 only': BLUE, '+ all eight ratios': ORANGE, '+ microbiota panel': GREEN}
MLAB = {'Age, sex, BMI': 'Age, sex, BMI', '+ R1 only': '+ R1 (ether-lipid ratio)',
        '+ all eight ratios': '+ all eight ratios', '+ microbiota panel': '+ CV microbiota panel'}


def roc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]
    o = np.argsort(-s); y = y[o]
    return np.r_[0, np.cumsum(1 - y) / (1 - y).sum(), 1], np.r_[0, np.cumsum(y) / y.sum(), 1]


def pan(ax, l, x=-0.16, y=1.045):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)


fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.32, left=0.095, right=0.975, top=0.945, bottom=0.15)
for cell, en, L in [(gs[0, 0], 'CKM stage 2 or higher', 'A'), (gs[0, 1], 'CKM stage 3 or 4', 'B'),
                    (gs[1, 0], 'Elevated triglycerides', 'C')]:
    ax = fig.add_subplot(cell); pan(ax, L)
    y = np.asarray(END[en], float)
    for mn, cols in MODELS.items():
        a_, oof = cv_auc(y, cols, 20261030)
        ref = float(SN[(SN.endpoint == en) & (SN.model == mn)].auc.iloc[0])
        assert abs(a_ - ref) < 1e-9, (en, mn, a_, ref)
        fp, tp = roc(y, oof)
        ax.plot(fp, tp, lw=1.3, color=MCOL[mn], label=f'{MLAB[mn]} — {a_:.3f}')
    ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color=GRID)
    ev = int(SN[(SN.endpoint == en)].n_events.iloc[0])
    ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity')
    ax.set_title(f'{en} ({ev} events of {n743})', fontsize=7.6, color=INK, pad=4)
    ax.legend(frameon=False, fontsize=6.0, loc='lower right', labelcolor=MUTED, handlelength=1.4)
    clean(ax)

d = fig.add_subplot(gs[1, 1]); pan(d, 'D')
order_en = ['CKM stage 2 or higher', 'CKM stage 3 or 4', 'Diabetes', 'Elevated triglycerides']
order_m = ['+ R1 only', '+ all eight ratios', '+ microbiota panel']
rows = []
for en in order_en:
    for mn in order_m:
        if mn == '+ microbiota panel':
            r = SW[(SW.endpoint == en) & (SW.model == mn)]
        else:
            r = SN[(SN.endpoint == en) & (SN.model == mn)]
        rows.append((en, mn, (float(r.delta.iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0])) if len(r) else None))
ytick, ylab = [], []
yy = 0.0
blocks = []
for i, en in enumerate(order_en):
    top = yy
    for mn in order_m:
        r = [x for x in rows if x[0] == en and x[1] == mn][0][2]
        if r is not None:
            dl, lo, hi = r
            d.plot([lo, hi], [yy, yy], lw=1.1, color=MCOL[mn], zorder=2)
            d.plot([dl], [yy], 'o', ms=4.0, color=MCOL[mn] if lo > 0 else 'white', mec=MCOL[mn], mew=1.3, zorder=3)
        else:
            d.text(0.004, yy, 'CV panel: not estimated for this endpoint', fontsize=5.2, color=MUTED, va='center', ha='left')
        yy -= 1.0
    ytick.append((top + yy + 1.0) / 2); ylab.append(en.replace(' or higher', '\nor higher').replace('Elevated ', 'Elevated\n'))
    yy -= 0.7
d.axvline(0, color=GRAY, lw=0.9, ls='--')
d.set_yticks(ytick); d.set_yticklabels(ylab, fontsize=6.4)
d.set_ylim(yy + 0.2, 0.8)
d.set_xlim(-0.03, 0.19)
d.set_xlabel('Gain in area under the curve over age, sex and BMI\n(paired bootstrap, 95% CI)')
d.set_title('The eight ratios add discrimination for the\nstage-2 criteria, not for stage 3 or 4', fontsize=7.6, color=INK, pad=4)
h = [plt.Line2D([], [], color=MCOL[m], lw=1.6, label=MLAB[m]) for m in order_m]
d.legend(handles=h, frameon=False, fontsize=5.8, labelcolor=MUTED, handlelength=1.4,
         loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=1, borderpad=0.1)
d.text(0.0, -0.62, 'Filled markers: the 95% interval excludes zero.', transform=d.transAxes,
       ha='left', va='top', fontsize=5.6, color=MUTED)
clean(d)

plt.savefig(f'Figure5_{ST}.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('chk/F5_v65.png', dpi=190, bbox_inches='tight', pad_inches=0.10)
plt.close()
print(f'Figure5_{ST}.svg')
