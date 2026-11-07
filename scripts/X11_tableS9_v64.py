# -*- coding: utf-8 -*-
"""Table S9 的敏感度分析，在 v64 的分析集上重做      20261010_v64

v52 的 S9 是在 v45 那版「基因型為入選條件」的樣本上做的（n = 1,948／585／436），
管線早已改成 MB825／MB743／METAB，分期也換成大表 V10。這裡用現行的分析集與
現行的分期重做同一組敏感度檢定，模型規格與 X2 一致：
HC3 穩健標準誤，校正年齡、性別、BMI 與 16S 定序批次（METAB 模型不含批次）。

★ 20261010 使用者裁示：不做 dx_queried == 1 的限制。分期以大表為準，
  大表寫幾期就是幾期，不存在「沒有診斷」這一類人，該切法的前提不成立。
  v52 版 S9 裡的 dx_queried 列一併作廢。

  本表只保留一個真正的敏感度問題：用藥資料的可得性 —— 把 med_* 缺值
  當 0，與只看確實有用藥資料者，兩種做法的對照。
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import stats

from stage_v10 import master_stage

UP = '/mnt/user-data/uploads/CKM+microbiome'
ST = '20260911_v54'                      # 與管線其餘輸出同一個內部戳記


def rint(v):
    a = np.asarray(v, float); ok = np.isfinite(a); o = np.full(len(a), np.nan)
    o[ok] = stats.norm.ppf((stats.rankdata(a[ok]) - 0.5) / ok.sum())
    return o


def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))


def fit(y, x, covs):
    """covs：欄位清單（已是數值陣列的 list）。回傳 x 的 beta/se/p/n。"""
    cols = [np.asarray(x, float)] + [np.asarray(c, float) for c in covs]
    M = np.column_stack(cols)
    ok = np.isfinite(M).all(1) & np.isfinite(np.asarray(y, float))
    if ok.sum() < 30:
        return np.nan, np.nan, np.nan, int(ok.sum())
    X = np.column_stack([np.ones(ok.sum()), M[ok]])
    b, se = hc3(X, np.asarray(y, float)[ok])
    z = b[1] / se[1]
    return b[1], se[1], 2 * stats.norm.sf(abs(z)), int(ok.sum())


# ------------------------------------------------------------------ 資料
PHE = pd.read_csv(f'pheno_all_{ST}.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
PHE['R1_int'] = rint(PHE.R1_PCae_lysoPC_z.values)
SC = pd.read_csv(f'panel_scores_{ST}.csv'); SC['subject_id'] = SC.subject_id.astype(str)

A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
A['CKM5'] = master_stage(A, 'X11 MB825')
C = pd.read_csv(f'{UP}/CKM_clinical_for_microbiome_20260819_V6.csv', low_memory=False)
C['subject_id'] = C.subject_id.astype(str)
C = C.sort_values('visit_no').drop_duplicates('subject_id').copy()
C['CKM5'] = master_stage(C, 'X11 clinical')

MED = [c for c in A.columns if c.startswith('med_')]
KEY = [c for c in ['med_Statin', 'med_Metformin', 'med_PPARalpha_Fibrate', 'med_SGLT2i',
                   'med_GLP1_RA', 'med_Insulin', 'med_Ezetimibe', 'med_PPARgamma_TZD']
       if c in A.columns]

A = A.merge(SC, on='subject_id', how='left').merge(
    PHE[['subject_id', 'R1_int']], on='subject_id', how='left')
A['panel_z'] = (A.panel_oof - A.panel_oof.mean()) / A.panel_oof.std()
A['prior_z'] = (A.score_prior - A.score_prior.mean()) / A.score_prior.std()
A['has_med'] = pd.to_numeric(A[MED[0]], errors='coerce').notna()
num = lambda d, c: pd.to_numeric(d[c], errors='coerce').values
cov_mb = lambda d: [num(d, 'age'), num(d, 'sex'), num(d, 'BMI')] + \
                   [(d.batch.astype(str) == b).astype(float).values
                    for b in sorted(d.batch.astype(str).unique())[1:]]

CL = C.merge(PHE[['subject_id', 'R1_int']], on='subject_id', how='inner')
cov_cl = lambda d: [num(d, 'age'), num(d, 'sex'), num(d, 'BMI')]

rows = []


def add(analysis, restriction, y, x, covs):
    b, se, p, n = fit(y, x, covs)
    rows.append(dict(analysis=analysis, restriction=restriction,
                     beta=b, se=se, p=p, n=n))


# ---- 主估計（全體），作為下面用藥敏感度的參照
for nm, col, tf in [('R1 -> CKM stage', 'CKM5', rint),
                    ('R1 -> Diabetes', 'C_DM', lambda v: np.asarray(v, float)),
                    ('R1 -> log(TG)', 'lab_triglyceride', lambda v: rint(np.log(
                        pd.to_numeric(v, errors='coerce').astype(float)))),
                    ('R1 -> HbA1c', 'lab_HbA1c', rint)]:
    add(nm, 'whole cohort', tf(CL[col].values), CL.R1_int.values, cov_cl(CL))

for nm, col, tf in [('Microbiome panel -> CKM stage', 'CKM5', rint),
                    ('Microbiome panel -> log(TG)', 'lab_triglyceride',
                     lambda v: rint(np.log(pd.to_numeric(v, errors='coerce').astype(float))))]:
    add(nm, 'whole cohort', tf(A[col].values), A.panel_z.values, cov_mb(A))
d = A[A.R1_int.notna()]
add('Microbiome panel -> PC ae/lysoPC', 'whole cohort',
    d.R1_int.values, d.panel_z.values, cov_mb(d))

# ---- 用藥校正：缺值當 0 vs 只看有用藥資料者
for nm, drugs in [('panel -> PC ae/lysoPC + 8 drugs', KEY),
                  ('panel -> PC ae/lysoPC + 28 drugs', MED)]:
    d0 = A[A.R1_int.notna()]
    M0 = [np.nan_to_num(num(d0, c)) for c in drugs]
    add(nm, 'missing coded 0', d0.R1_int.values, d0.panel_z.values, cov_mb(d0) + M0)
    d1 = d0[d0.has_med]
    M1 = [num(d1, c) for c in drugs]
    add(nm, 'restricted to those with medication data',
        d1.R1_int.values, d1.panel_z.values, cov_mb(d1) + M1)
d0 = A.copy()
M0 = [np.nan_to_num(num(d0, c)) for c in MED]
y0 = rint(np.log(pd.to_numeric(d0.lab_triglyceride, errors='coerce').astype(float)))
add('panel -> log(TG) + 28 drugs', 'missing coded 0', y0, d0.panel_z.values, cov_mb(d0) + M0)
d1 = d0[d0.has_med]
M1 = [num(d1, c) for c in MED]
y1 = rint(np.log(pd.to_numeric(d1.lab_triglyceride, errors='coerce').astype(float)))
add('panel -> log(TG) + 28 drugs', 'restricted to those with medication data',
    y1, d1.panel_z.values, cov_mb(d1) + M1)

S9 = pd.DataFrame(rows)
S9.to_csv(f'TableS9_subgroup_sensitivity_{ST}.csv', index=False)
pd.set_option('display.width', 200)
print(S9.to_string(index=False))
print(f'\n寫出 TableS9_subgroup_sensitivity_{ST}.csv（{len(S9)} 列）')
