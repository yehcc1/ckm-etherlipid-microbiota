# -*- coding: utf-8 -*-
"""缺分期者被當成「未達切點」的錯誤修正 — 20261107_v66。

外審（v66 審稿意見 S35–S39 stage denominators）指出：R1 對五級分期 n = 3,367，二元切點 n = 3,419。
查核結果：代謝體 ∩ 臨床 3,679 人中 271 人無 CKM 分期；`(stage >= cut).astype(float)` 會把 NaN 變成 0，
於是二元切點模型把無分期者當成非事件。受影響：
  1. Table S8（v66 為 Table S35）的「CKM stage >= 3」八列（X2 第 487 行）
  2. X21 A：八比值區塊對 >=1..>=4 的概似比檢定與各比值係數（TableSp／TableSo）
  3. X22：疾病端八比值軸的訓練樣本（TRAIN 中無分期者被當成非事件）
微生物樣本（825／743）每個人都有分期，不受影響。

本腳本：(1) 先用原寫法重現舊值，確認與現有表逐位相同（證明複製正確）；(2) 改成無分期設為 NaN 後重算；
(3) 輸出新表與新舊對照表，並寫 numbers 檔供正文 token 用。X21／X22 原始碼同步修正後重跑。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
exec(open('X21_eight_ratios_stage_v65.py').read().split('def hc3')[0])   # 同一份資料載入（含 master_stage）

def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty(len(p)); run = 1.0
    for i, idx in enumerate(o[::-1]):
        run = min(run, p[idx] * len(p) / (len(p) - i)); q[idx] = run
    return q

for r in RAT: PHE[r + '_int'] = rint(PHE[r].values)
ph = PHE.set_index('subject_id')
age_cl = pd.to_numeric(cl.age, errors='coerce'); sex_cl = pd.to_numeric(cl.sex, errors='coerce')
bmi_cl = pd.to_numeric(cl.BMI, errors='coerce')
def fit_cl(y, x, idx, binary):
    Cv = np.column_stack([np.ones(len(idx)), age_cl.reindex(idx).values, sex_cl.reindex(idx).values,
                          bmi_cl.reindex(idx).values])
    X = np.column_stack([np.asarray(x, float), Cv]); yv = np.asarray(y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(yv)
    if binary:
        r = sm.GLM(yv[ok], X[ok], family=sm.families.Binomial()).fit(cov_type='HC3')
        return r.params[0], r.bse[0], r.pvalues[0], int(ok.sum())
    b, se = hc3(X[ok], yv[ok])
    return b[0], se[0], 2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - X.shape[1]), int(ok.sum())

stage = pd.to_numeric(cl.CKM5, errors='coerce').reindex(SM_).values
OLD = pd.read_csv('out/TableS8_ratio_outcome_20261030_v65.csv')
y_bug = (stage >= 3).astype(float)
y_fix = np.where(np.isnan(stage), np.nan, (stage >= 3).astype(float))
print(f'代謝體∩臨床 {len(SM_)} 人，無分期 {int(np.isnan(stage).sum())} 人')

# (1) 重現舊值
for r in RAT:
    b, se, p, n = fit_cl(y_bug, ph[r + '_int'].reindex(SM_).values, SM_, True)
    o = OLD[(OLD.ratio == r.split('_')[0]) & (OLD.outcome == 'CKM stage >= 3')].iloc[0]
    assert abs(b - o.beta) < 1e-9 and n == o.n, f'{r}: 重現失敗 {b} vs {o.beta}, n {n} vs {o.n}'
print('舊寫法完全重現 Table S8 的 stage >= 3 八列（含 n = 3,419）')

# (2) 修正後重算，整個家族重做 BH
NEW = OLD.copy()
for r in RAT:
    b, se, p, n = fit_cl(y_fix, ph[r + '_int'].reindex(SM_).values, SM_, True)
    i = NEW[(NEW.ratio == r.split('_')[0]) & (NEW.outcome == 'CKM stage >= 3')].index[0]
    NEW.loc[i, ['beta', 'se', 'p', 'n']] = [b, se, p, n]
NEW['q'] = bh(NEW.p.values)
NEW = NEW.sort_values('p')
NEW.to_csv('out/TableS8_ratio_outcome_20261107_v66.csv', index=False)
cmp = OLD.merge(NEW, on=['ratio', 'outcome'], suffixes=('_old', '_new'))
cmp = cmp[['ratio', 'outcome', 'beta_old', 'beta_new', 'p_old', 'p_new', 'q_old', 'q_new', 'n_old', 'n_new']]
cmp.to_csv('chk/S8_old_vs_new_20261107.csv', index=False)
pd.set_option('display.width', 220)
print(cmp[cmp.outcome == 'CKM stage >= 3'].round(5).to_string(index=False))
moved = cmp[(cmp.outcome != 'CKM stage >= 3') & ((cmp.q_old - cmp.q_new).abs() > 1e-12)]
print(f'\n其他結局：β、P、n 全部不變；q 因 BH 家族重排而變動的列 {len(moved)}')
print(moved[['ratio', 'outcome', 'q_old', 'q_new']].to_string(index=False))
