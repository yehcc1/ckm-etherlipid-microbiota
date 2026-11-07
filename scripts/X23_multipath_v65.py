# -*- coding: utf-8 -*-
"""多條通路一起看：菌相經由八個比值到 CKM 的總間接效果      20261030_v65

先前我逐條測「菌相 → R3」「菌相 → R7」，各自不顯著就說這條路斷了。
那是錯的看法：如果菌相是經由多條代謝物通路的綜合結果到達疾病，每一條單獨
都可以很小，加起來才有份量，而且逐條做 BH 會把這種分散的訊號整個殺掉。

這支腳本改成三種多通路的做法：

  A  綜合檢定：八個比值當一個區塊，和菌相分數有沒有關聯（概似比／F 檢定）。
     這回答「代謝體裡有沒有任何一個組合帶著菌相的訊號」，不限定是哪一條。

  B  菌相關聯的代謝體軸：取 A 的線性組合（菌相在八個比值上的投影），
     再問這條軸和 CKM 各分期切點有沒有關聯。這和 X22 的方向相反——
     X22 是從疾病端建軸，這裡是從菌相端建軸。

  C  平行多重中介：八個比值同時當中介，算每一條的 a_j × b_j 與**總間接效果**，
     bootstrap 取信賴區間。總間接效果才是「多條通路的綜合結果」的正確統計量。

主要暴露依裁示改為事前三菌屬分數；CV 面板一併報為次要。
共變數：年齡、性別、BMI、定序批次，與主管線相同。
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
SRC, ST = '20260911_v54', '20261030_v65'
B_BOOT = 2000
RAT = ['R1_PCae_lysoPC_z', 'R2_PCae362_361_z', 'R3_SM_over_PC_z', 'R4_SM240_160_z',
       'R5_lysoPC204_PCaa384_z', 'R6_PCaa384_383_z', 'R7_C16_C0_z', 'R8_C2_C0_z']
SHORT = [r.split('_')[0] for r in RAT]

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
A['CKM5'] = master_stage(A, 'X23', quiet=True)
mb = A.set_index('subject_id'); S825 = list(A.subject_id)
S743 = [s for s in S825 if s in set(PHE.subject_id)]
ph = PHE.set_index('subject_id')
PAN = pd.read_csv(f'panel_scores_{SRC}.csv', index_col=0); PAN.index = PAN.index.astype(str)

_bc = mb.batch.value_counts(); _small = set(_bc[_bc < 5].index)
BAT = pd.get_dummies(mb.batch.where(~mb.batch.isin(_small), 'BatchOther'),
                     drop_first=True).astype(float)
COV = np.column_stack([pd.to_numeric(mb.age, errors='coerce').reindex(S743).values,
                       pd.to_numeric(mb.sex, errors='coerce').reindex(S743).values,
                       pd.to_numeric(mb.BMI, errors='coerce').reindex(S743).values,
                       BAT.reindex(S743).values])
R = np.column_stack([rint(ph[r].reindex(S743).values) for r in RAT])
EXPO = {'Pre-specified score': rint(PAN.score_prior.reindex(S743).values),
        'CV panel': rint(PAN.panel_oof.reindex(S743).values)}
stage = pd.to_numeric(mb.CKM5, errors='coerce').reindex(S743).values
OUT = {'CKM stage (5-level)': ('lin', rint(stage)),
       'CKM stage >= 2': ('bin', (stage >= 2).astype(float)),
       'CKM stage >= 3': ('bin', (stage >= 3).astype(float)),
       'CKM stage >= 4': ('bin', (stage >= 4).astype(float))}


def rank_clean(X):
    keep = np.ones(X.shape[1], bool)
    for j in range(X.shape[1]):
        if np.nanstd(X[:, j]) < 1e-12: keep[j] = False
    return X[:, keep]


def fit_one(y, X, kind):
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    Xd = sm.add_constant(rank_clean(X[ok]), has_constant='add')
    m = (sm.GLM(y[ok], Xd, family=sm.families.Binomial()).fit() if kind == 'bin'
         else sm.OLS(y[ok], Xd).fit())
    return m, ok


# =============================================== A. 綜合檢定
rows = []
for xn, xv in EXPO.items():
    m1, ok1 = fit_one(xv, np.column_stack([R, COV]), 'lin')
    m0, ok0 = fit_one(xv, COV, 'lin')
    n = int(ok1.sum())
    rss1, rss0 = m1.ssr, m0.ssr
    k = R.shape[1]
    Fst = ((rss0 - rss1) / k) / (rss1 / (n - m1.df_model - 1))
    pF = stats.f.sf(Fst, k, n - m1.df_model - 1)
    r2 = (rss0 - rss1) / rss0
    rows.append(dict(exposure=xn, test='eight ratios as a block -> microbiota score',
                     stat=Fst, df1=k, df2=int(n - m1.df_model - 1), p=pF,
                     partial_r2=r2, n=n))
OMNI = pd.DataFrame(rows)
print('── A. 八個比值當一個區塊，對菌相分數 ──')
print(OMNI.round(5).to_string(index=False)); print()

# =============================================== B. 菌相關聯的代謝體軸
axrows = []
for xn, xv in EXPO.items():
    m1, ok1 = fit_one(xv, np.column_stack([R, COV]), 'lin')
    w = m1.params[1:1 + R.shape[1]]
    ax = pd.Series(rint(R @ w), index=S743)
    for en, (kind, y) in OUT.items():
        X = np.column_stack([ax.values, COV])
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        Xd = sm.add_constant(rank_clean(X[ok]), has_constant='add')
        m = (sm.GLM(y[ok], Xd, family=sm.families.Binomial()).fit() if kind == 'bin'
             else sm.OLS(y[ok], Xd).fit(cov_type='HC3'))
        axrows.append(dict(exposure=xn, axis='microbiota-associated metabolite axis',
                           outcome=en, beta=m.params[1], se=m.bse[1], p=m.pvalues[1],
                           n=int(ok.sum())))
    for j, s_ in enumerate(SHORT):
        axrows.append(dict(exposure=xn, axis='weight', outcome=s_, beta=w[j],
                           se=m1.bse[1 + j], p=m1.pvalues[1 + j], n=int(ok1.sum())))
AXIS = pd.DataFrame(axrows)
print('── B. 菌相關聯的代謝體軸 → CKM ──')
print(AXIS[AXIS.axis != 'weight'].round(4).to_string(index=False))
print('\n  軸的權重')
print(AXIS[AXIS.axis == 'weight'].pivot(index='outcome', columns='exposure',
                                        values='beta').round(3).to_string()); print()


# =============================================== C. 平行多重中介
def paths(idx_rows, xv, y, kind):
    Rr, Cc, xx, yy = R[idx_rows], COV[idx_rows], xv[idx_rows], y[idx_rows]
    a = np.full(R.shape[1], np.nan)
    for j in range(R.shape[1]):
        X = np.column_stack([xx, Cc]); ok = np.isfinite(X).all(1) & np.isfinite(Rr[:, j])
        if ok.sum() < 60: continue
        m = sm.OLS(Rr[ok, j], sm.add_constant(rank_clean(X[ok]), has_constant='add')).fit()
        a[j] = m.params[1]
    X = np.column_stack([xx, Rr, Cc]); ok = np.isfinite(X).all(1) & np.isfinite(yy)
    if ok.sum() < 80: return a, np.full(R.shape[1], np.nan), np.nan, np.nan
    Xd = sm.add_constant(rank_clean(X[ok]), has_constant='add')
    m = (sm.GLM(yy[ok], Xd, family=sm.families.Binomial()).fit() if kind == 'bin'
         else sm.OLS(yy[ok], Xd).fit())
    cprime = m.params[1]; b = m.params[2:2 + R.shape[1]]
    Xc = np.column_stack([xx, Cc]); okc = np.isfinite(Xc).all(1) & np.isfinite(yy)
    Xdc = sm.add_constant(rank_clean(Xc[okc]), has_constant='add')
    mc = (sm.GLM(yy[okc], Xdc, family=sm.families.Binomial()).fit() if kind == 'bin'
          else sm.OLS(yy[okc], Xdc).fit())
    return a, b, mc.params[1], cprime


rng = np.random.default_rng(20261030)
med = []
allrows = np.arange(len(S743))
for xn, xv in EXPO.items():
    for en, (kind, y) in OUT.items():
        a, b, c, cp = paths(allrows, xv, y, kind)
        ind = np.nansum(a * b); tot = ind + cp
        bi, bc = [], []
        for _ in range(B_BOOT):
            take = rng.integers(0, len(S743), len(S743))
            try:
                a_, b_, c_, cp_ = paths(take, xv, y, kind)
            except Exception:
                continue
            v = np.nansum(a_ * b_)
            if np.isfinite(v): bi.append(v)
            if np.isfinite(v) and np.isfinite(cp_) and (v + cp_) != 0:
                bc.append(v / (v + cp_))
        lo, hi = np.percentile(bi, [2.5, 97.5])
        plo, phi = (np.percentile(bc, [2.5, 97.5]) if len(bc) > 100 else (np.nan, np.nan))
        med.append(dict(exposure=xn, outcome=en, c_total=c, c_prime=cp,
                        total_indirect=ind, ind_lo=lo, ind_hi=hi,
                        prop_mediated=ind / tot if tot else np.nan,
                        prop_lo=plo, prop_hi=phi,
                        pr_ind_cross0=float(np.mean(np.sign(bi) != np.sign(np.median(bi)))),
                        n_boot=len(bi),
                        **{f'ind_{s_}': a[j] * b[j] for j, s_ in enumerate(SHORT)}))
        print(f'  {xn:20s} {en:22s} 總間接 {ind:+.4f} ({lo:+.4f}, {hi:+.4f})  c′ {cp:+.4f}')
MED = pd.DataFrame(med)
OMNI.to_csv(f'TableSj_omnibus_{ST}.csv', index=False)
AXIS.to_csv(f'TableSi_microbiota_metabolite_axis_{ST}.csv', index=False)
MED.to_csv(f'TableSh_parallel_mediation_{ST}.csv', index=False)
print('\n── C. 平行多重中介（八個比值同時當中介）──')
cols = ['exposure', 'outcome', 'c_total', 'c_prime', 'total_indirect', 'ind_lo', 'ind_hi',
        'prop_mediated']
print(MED[cols].round(4).to_string(index=False))
print('\n  每一條通路的 a_j x b_j（事前分數）')
print(MED[MED.exposure == 'Pre-specified score'].set_index('outcome')[
    [f'ind_{s_}' for s_ in SHORT]].round(4).to_string())
print(f'\n寫出 TableSj_omnibus / TableSi_microbiota_metabolite_axis / '
      f'TableSh_parallel_mediation（_{ST}.csv）')
