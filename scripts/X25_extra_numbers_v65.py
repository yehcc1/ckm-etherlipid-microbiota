# -*- coding: utf-8 -*-
"""正文需要、但原 numbers 檔只有 CV 面板版本的數字，改以事前分數重算    20261030_v65

  cls_pre_*          五個類別層級表型對事前分數（直接讀 TableS4）
  spc_pre_*          37 個 PC ae 分子比值對事前分數：q<0.05 數、同向數、I²；絕對濃度版本；
                     與 13 個 lysoPC 分子的 Mann–Whitney
  carbon_pre_* / doublebond_pre_*   事前分數的碳數／雙鍵梯度，與 X2 同一個 HC3 聯合模型
全部讀自 TableS4 / TableS5，沒有手打。
"""
import numpy as np, pandas as pd
from scipy import stats
ST = '20261030_v65'
rows = []
add = lambda k, v, src: rows.append(dict(key=k, value=v, source=src))


def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))


S4 = pd.read_csv(f'out/TableS4_class_specificity_{ST}.csv')
key = {'PC ae / PC aa': 'pc_ae_over_pc_aa', 'Total PC ae': 'total_pc_ae', 'Total PC aa': 'total_pc_aa',
       'Total lysoPC': 'total_lysopc', 'Short/long PC ae': 'short_long_pc_ae'}
for _, r in S4.iterrows():
    tag = 'pre' if r.score.startswith('Pre') else 'cv'
    add(f'cls_{tag}_{key[r.phenotype]}_beta', r.beta, 'TableS4'); add(f'cls_{tag}_{key[r.phenotype]}_p', r.p, 'TableS4')
    add(f'cls_{tag}_{key[r.phenotype]}_n', r.n, 'TableS4')

S5 = pd.read_csv(f'out/TableS5_species_{ST}.csv')
for sc, tag in [('Pre-specified score', 'pre'), ('CV panel', 'cv')]:
    d1 = S5[(S5.group == 'PC ae / total lysoPC') & (S5.score == sc)]
    add(f'spc_{tag}_n', len(d1), 'TableS5'); add(f'spc_{tag}_q05', int((d1.q < 0.05).sum()), 'TableS5')
    add(f'spc_{tag}_same_direction', int((np.sign(d1.beta) == np.sign(d1.beta.median())).sum()), 'TableS5')
    w = 1 / d1.se.values ** 2; mu = np.sum(w * d1.beta.values) / w.sum()
    Q = np.sum(w * (d1.beta.values - mu) ** 2); add(f'spc_{tag}_I2', max(0.0, (Q - (len(d1) - 1)) / Q), 'TableS5')
    cn = d1.species.str.extract(r'C(\d+):(\d+)').astype(float)
    Xg = np.column_stack([np.ones(len(d1)), cn[0].values, cn[1].values])
    bb, ss = hc3(Xg, d1.beta.values); df = len(d1) - 3
    add(f'carbon_{tag}_beta', bb[1], 'TableS5'); add(f'carbon_{tag}_p', 2 * stats.t.sf(abs(bb[1] / ss[1]), df), 'TableS5')
    add(f'doublebond_{tag}_beta', bb[2], 'TableS5'); add(f'doublebond_{tag}_p', 2 * stats.t.sf(abs(bb[2] / ss[2]), df), 'TableS5')
    ab = S5[(S5.group == 'PC ae absolute') & (S5.score == sc)]
    ly = S5[(S5.group == 'lysoPC absolute') & (S5.score == sc)]
    add(f'spc_{tag}_abs_q05', int((ab.q < 0.05).sum()), 'TableS5')
    add(f'spc_{tag}_abs_same_direction', int((np.sign(ab.beta) == np.sign(ab.beta.median())).sum()), 'TableS5')
    add(f'spc_{tag}_lyso_n', len(ly), 'TableS5'); add(f'spc_{tag}_lyso_q05', int((ly.q < 0.05).sum()), 'TableS5')
    add(f'spc_{tag}_mw_p', float(stats.mannwhitneyu(ab.beta.values, ly.beta.values, alternative='two-sided').pvalue), 'TableS5')
# 與 X2 的 CV 面板數字核對
N = pd.read_csv(f'numbers_v65_all_{ST}.csv').drop_duplicates('key', keep='last').set_index('key').value
E = pd.DataFrame(rows).set_index('key').value
assert abs(float(E['carbon_cv_beta']) - float(N['carbon_gradient_beta'])) < 1e-9
assert abs(float(E['carbon_cv_p']) - float(N['carbon_gradient_p'])) < 1e-9
assert int(E['spc_cv_q05']) == int(float(N['n_species_q05']))
pd.DataFrame(rows).to_csv(f'numbers_extra_{ST}.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
