# -*- coding: utf-8 -*-
"""ΔAUC 的配對 bootstrap      20261030_v65

先前我用「兩個模型的 AUC 信賴區間重疊」來說面板沒有增益，那是錯的推論方式：
兩個模型在同一批人身上評分，AUC 高度相關，分別看各自的區間會嚴重低估增益的
精確度。正確的做法是在每一次 bootstrap 重抽裡算出兩個模型的 AUC 之差，再看
那個差值分布。

這支腳本在每一次重抽裡跑完整的 nested CV（含菌屬選擇），同時存下四個模型的
AUC，最後報 ΔAUC 的點估計、95% 百分位區間，以及 ΔAUC <= 0 的比例。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from nested_core_v65 import (S743, END, MODELS, auc, nested_predictions, ST, SEED_OUTER)

B = 500
BASE = 'Age, sex, BMI'
rng = np.random.default_rng(20261030)

# 點估計
point = {}
for en, y in END.items():
    preds, yv, _ = nested_predictions(S743, y, 'p01')
    point[en] = {m: auc(yv, preds[m]) for m in MODELS}

raw = {(en, m): [] for en in END for m in MODELS}
for b in range(B):
    take = list(pd.Series(S743).sample(len(S743), replace=True,
                                       random_state=int(rng.integers(1e9))))
    for en, y in END.items():
        try:
            preds, yv, _ = nested_predictions(take, y, 'p01', seed=SEED_OUTER + b)
        except Exception:
            continue
        for m in MODELS:
            raw[(en, m)].append(auc(yv, preds[m]))
    if (b + 1) % 100 == 0:
        print(f'  {b + 1}/{B}')

# 額外對比：在「臨床＋醚脂比值」之上再加菌相面板，還會不會動。
# 這是「菌相的臨床訊息是不是已經被脂質讀數吸收掉」的直接檢定。
EXTRA = ('+ ether-lipid ratio', '+ microbiota panel and ether-lipid ratio')
rows = []
for en in END:
    r_v = np.array(raw[(en, EXTRA[0])], float); b_v = np.array(raw[(en, EXTRA[1])], float)
    n_ = min(len(r_v), len(b_v)); dd = b_v[:n_] - r_v[:n_]; dd = dd[np.isfinite(dd)]
    rows.append(dict(endpoint=en, model='panel on top of ether-lipid ratio',
                     auc_base=point[en][EXTRA[0]], auc_model=point[en][EXTRA[1]],
                     delta=point[en][EXTRA[1]] - point[en][EXTRA[0]],
                     delta_boot_mean=float(dd.mean()),
                     lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5)),
                     pr_delta_le_0=float((dd <= 0).mean()), n_boot=int(len(dd))))
for en in END:
    base_v = np.array(raw[(en, BASE)], float)
    for m in MODELS:
        if m == BASE: continue
        v = np.array(raw[(en, m)], float)
        n = min(len(v), len(base_v))
        d = v[:n] - base_v[:n]
        d = d[np.isfinite(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(endpoint=en, model=m,
                         auc_base=point[en][BASE], auc_model=point[en][m],
                         delta=point[en][m] - point[en][BASE],
                         delta_boot_mean=float(d.mean()), lo=float(lo), hi=float(hi),
                         pr_delta_le_0=float((d <= 0).mean()), n_boot=int(len(d))))
D = pd.DataFrame(rows)
D = D.sort_values(['endpoint','model'])
D.to_csv(f'TableSw_delta_auc_{ST}.csv', index=False)
print('\n配對 ΔAUC（每次重抽裡同一批人、同一條 pipeline，兩個模型相減）')
print(D[['endpoint', 'model', 'auc_base', 'auc_model', 'delta', 'lo', 'hi',
         'pr_delta_le_0']].round(4).to_string(index=False))
print(f'\n寫出 TableSw_delta_auc_{ST}.csv')
