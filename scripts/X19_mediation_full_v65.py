# -*- coding: utf-8 -*-
"""完整的 a／b／c／c′ 與「菌相有沒有獨立於脂質的直接路徑」      20261030_v65

使用者提的結構是：菌相 → 血中代謝物 → 疾病；而且**菌相本身不直接致病**。
要支持這個結構，關鍵不是間接效果有多大，而是**直接效果 c′ 有多小**——
扣掉脂質讀數之後，菌相還剩下多少對結局的關聯。原本只報了間接效果佔比，
沒有把 c′ 單獨拿出來檢定，所以這個結構從來沒有被直接驗證過。

這支腳本對每一個結局算：

  a   菌相 → 醚脂比值
  b   醚脂比值 → 結局（模型裡同時放菌相）
  c   菌相 → 結局（總效果）
  c′  菌相 → 結局（模型裡同時放醚脂比值）＝ 直接效果
  indirect = a × b，proportion = a × b / c

共變數：年齡、性別、BMI、定序批次（與主要模型相同）。
連續結局用 OLS，糖尿病用 logistic。信賴區間用 2,000 次 bootstrap。

同時報判別力的另一個對比：在「臨床＋醚脂比值」之上再加菌相面板，
AUC 還會不會動。那是「菌相的臨床訊息是不是已經被脂質讀數吸收掉」的直接檢定。
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
SRC = '20260911_v54'
ST = '20261030_v65'
B = 2000

rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))

PHE = pd.read_csv(f'pheno_all_{SRC}.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
from stage_v10 import master_stage
A['CKM5'] = master_stage(A, 'X19', quiet=True)
mb = A.set_index('subject_id')
S825 = list(A.subject_id)
S743 = [s for s in S825 if s in set(PHE.subject_id)]

PAN = pd.read_csv(f'panel_scores_{SRC}.csv', index_col=0); PAN.index = PAN.index.astype(str)
ph = PHE.set_index('subject_id')
R1 = pd.Series(rint(ph.R1_PCae_lysoPC_z.values), index=ph.index)

cov = pd.DataFrame({'age': pd.to_numeric(mb.age, errors='coerce'),
                    'sex': pd.to_numeric(mb.sex, errors='coerce'),
                    'bmi': pd.to_numeric(mb.BMI, errors='coerce')})
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float)
COV = pd.concat([cov, BAT], axis=1)

OUT = {
    'log(triglycerides)': ('lin', pd.Series(rint(np.log(pd.to_numeric(
        mb.lab_triglyceride, errors='coerce').values)), index=mb.index)),
    'HbA1c': ('lin', pd.Series(rint(pd.to_numeric(mb.lab_HbA1c, errors='coerce').values),
                               index=mb.index)),
    'CKM stage (5-level)': ('lin', pd.Series(rint(pd.to_numeric(mb.CKM5, errors='coerce').values),
                                             index=mb.index)),
    'Diabetes': ('bin', pd.to_numeric(mb.C_DM, errors='coerce')),
}
EXPO = {'CV panel': PAN.panel_oof, 'Pre-specified score': PAN.score_prior}


def drop_constant(X):
    """bootstrap 重抽之後，人數很少的批次（Batch12 只有 1 人）可能整欄變成 0，
    設計矩陣就會降秩。把常數欄拿掉；暴露永遠在最前面，係數位置不受影響。"""
    keep = np.ones(X.shape[1], bool)
    for j in range(2, X.shape[1]):
        if np.nanstd(X[:, j]) < 1e-12: keep[j] = False
    return X[:, keep]


def fit(y, X, kind):
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() < 50: return np.nan, np.nan, 0
    Xd = sm.add_constant(drop_constant(X[ok]), has_constant='add')
    if kind == 'bin':
        m = sm.GLM(y[ok], Xd, family=sm.families.Binomial()).fit()
    else:
        m = sm.OLS(y[ok], Xd).fit(cov_type='HC3')
    return m.params[1], m.bse[1], int(ok.sum())


def paths(idx, xs, yv, kind):
    """回傳 a, b, c, cprime。idx 是要用的人；xs 是暴露；yv 是結局。"""
    C = COV.reindex(idx).values.astype(float)
    x = np.asarray(xs.reindex(idx).values, float)
    r = np.asarray(R1.reindex(idx).values, float)
    y = np.asarray(yv.reindex(idx).values, float)
    a, _, na = fit(r, np.column_stack([x, C]), 'lin')
    c, _, nc = fit(y, np.column_stack([x, C]), kind)
    Xboth = np.column_stack([x, r, C])
    ok = np.isfinite(Xboth).all(1) & np.isfinite(y)
    if ok.sum() < 50: return a, np.nan, c, np.nan, nc
    Xd = sm.add_constant(drop_constant(Xboth[ok]), has_constant='add')
    m = (sm.GLM(y[ok], Xd, family=sm.families.Binomial()).fit() if kind == 'bin'
         else sm.OLS(y[ok], Xd).fit(cov_type='HC3'))
    return a, m.params[2], c, m.params[1], int(ok.sum())


rows = []
rng = np.random.default_rng(20261030)
for en, (kind, yv) in OUT.items():
    for xn, xs in EXPO.items():
        a, b, c, cp, n = paths(S743, xs, yv, kind)
        ind = a * b
        prop = ind / c if np.isfinite(c) and c != 0 else np.nan
        bi, bp, bc = [], [], []
        for _ in range(B):
            take = [S743[i] for i in rng.integers(0, len(S743), len(S743))]
            try:
                a_, b_, c_, cp_, _ = paths(take, xs, yv, kind)
            except Exception:
                continue
            if not np.isfinite(a_ * b_) or not np.isfinite(cp_): continue
            bi.append(a_ * b_); bp.append(cp_)
            if np.isfinite(c_) and c_ != 0: bc.append(a_ * b_ / c_)
        q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) > 50 else (np.nan, np.nan)
        ilo, ihi = q(bi); plo, phi = q(bp); clo, chi = q(bc)
        rows.append(dict(outcome=en, exposure=xn, n=n, scale='log-odds' if kind == 'bin' else 'SD',
                         a=a, b=b, c=c, c_prime=cp,
                         indirect=ind, indirect_lo=ilo, indirect_hi=ihi,
                         c_prime_lo=plo, c_prime_hi=phi,
                         proportion=prop, proportion_lo=clo, proportion_hi=chi,
                         pr_cprime_cross0=float(np.mean(np.sign(bp) != np.sign(np.median(bp))))))
M = pd.DataFrame(rows)
M.to_csv(f'TableSv_mediation_full_{ST}.csv', index=False)

pd.set_option('display.width', 250)
print('完整路徑係數（a 菌相→比值；b 比值→結局|菌相；c 總效果；c′ 直接效果）\n')
for xn in EXPO:
    print(f'── {xn} ──')
    d = M[M.exposure == xn]
    print(d[['outcome', 'n', 'a', 'b', 'c', 'c_prime', 'c_prime_lo', 'c_prime_hi',
             'indirect', 'indirect_lo', 'indirect_hi',
             'proportion', 'proportion_lo', 'proportion_hi']].round(4).to_string(index=False))
    print()
print(f'寫出 TableSv_mediation_full_{ST}.csv')
