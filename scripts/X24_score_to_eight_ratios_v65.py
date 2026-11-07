# -*- coding: utf-8 -*-
"""兩種菌相分數對八個事前比值（Figure 2B、Table S33）         20261030_v65

與 X2 的穩健度階梯「Age, sex, batch, BMI」完全同一個模型：
  結局 = 管線的 _int 比值（在 3,688 人上做過一次 rank-INT）；
  暴露 = panel_scores（事前分數與 out-of-fold 面板，已在 743 人上標準化）；
  共變數 = 年齡、性別、BMI、定序批次（全部批次的虛擬變數，與 X2 相同）；
  HC3（槓桿值截在 0.999），t 分布 P。
R1 那一列必須與 TableS6 的 base model 完全一致（assert）。
原本的 TableSm 是臨時腳本做的（在 743 人內重做 rank-INT），R1 得 0.190 而不是 0.182；
本腳本取代它，讓 Figure 2B 與 Figure 2C 用同一把尺。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

UP = '/mnt/user-data/uploads/CKM+microbiome'
SRC, ST = '20260911_v54', '20261030_v65'
RAT = ['R1_PCae_lysoPC_z', 'R2_PCae362_361_z', 'R3_SM_over_PC_z', 'R4_SM240_160_z',
       'R5_lysoPC204_PCaa384_z', 'R6_PCaa384_383_z', 'R7_C16_C0_z', 'R8_C2_C0_z']


def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty(len(p)); run = 1.0
    for i, idx in enumerate(o[::-1]):
        run = min(run, p[idx] * len(p) / (len(p) - i)); q[idx] = run
    return q


rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))
PHE = pd.read_csv(f'pheno_all_{SRC}.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
for r in RAT: PHE[r + '_int'] = rint(PHE[r].values)          # rank-INT over all 3,688（與 X2 相同）
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
S825 = list(A.subject_id); S743 = [s for s in S825 if s in set(PHE.subject_id)]
mb = A.set_index('subject_id'); ph = PHE.set_index('subject_id')
PAN = pd.read_csv(f'out/panel_scores_{ST}.csv'); PAN['subject_id'] = PAN.subject_id.astype(str)
PAN = PAN.set_index('subject_id')
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float)
Cv = np.column_stack([np.ones(len(S743)), pd.to_numeric(mb.age, errors='coerce').reindex(S743).values,
                      pd.to_numeric(mb.sex, errors='coerce').reindex(S743).values,
                      pd.to_numeric(mb.BMI, errors='coerce').reindex(S743).values,
                      BAT.reindex(S743).values])
EXPO = {'Pre-specified score': PAN.score_prior.reindex(S743).values,
        'CV panel': PAN.panel_oof.reindex(S743).values}
rows = []
for ex, x in EXPO.items():
    for r in RAT:
        y = ph[r + '_int'].reindex(S743).values
        X = np.column_stack([x, Cv]); ok = np.isfinite(X).all(1) & np.isfinite(y)
        X, yv = X[ok], y[ok]
        keep = np.r_[True, True, X[:, 2:].std(0) > 1e-10]; X = X[:, keep]
        b, se = hc3(X, yv)
        p = 2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - X.shape[1])
        rows.append(dict(exposure=ex, ratio=r, beta=b[0], se=se[0], p=p, n=int(ok.sum())))
M = pd.DataFrame(rows)
M['q_across_8'] = np.nan
for ex, g in M.groupby('exposure'): M.loc[g.index, 'q_across_8'] = bh(g.p.values)

ROB = pd.read_csv(f'out/TableS6_robustness_{ST}.csv')
for ex in EXPO:
    ref = ROB[(ROB.Exposure == ex) & (ROB.Outcome == 'PC ae/lysoPC') & (ROB.Model == 'Age, sex, batch, BMI')].iloc[0]
    got = M[(M.exposure == ex) & (M.ratio == RAT[0])].iloc[0]
    assert abs(got.beta - ref.Beta) < 1e-9 and abs(got.p - ref.P) < 1e-12, (ex, got.beta, ref.Beta)
M.to_csv(f'TableSm_panel_to_eight_ratios_{ST}.csv', index=False)
pd.set_option('display.width', 200)
print(M.round(4).to_string(index=False))
print(f'\nR1 與 TableS6 base model 完全一致；寫出 TableSm_panel_to_eight_ratios_{ST}.csv')
