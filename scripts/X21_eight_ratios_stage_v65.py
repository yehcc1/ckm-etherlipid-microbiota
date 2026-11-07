# -*- coding: utf-8 -*-
"""八個比值一起看 CKM 分期，而不是只用 R1      20261030_v65

使用者指出：前一篇分期論文用的是八個特徵，我卻只用單一個 R1 去測 CKM 第三／第四期
就下結論，這是以偏概全，也是 AUC 一直上不來的原因。他是對的。

這支腳本改成把八個事前比值一起放進模型：

  A  每一個分期切點（>=1、>=2、>=3、>=4）上，八個比值當一個區塊做概似比檢定，
     同時報每一個比值自己的係數。大樣本（代謝體 ∩ 臨床）與 MB743 都做。
  B  判別力：年齡性別 BMI 當基準，加「單一 R1」與加「八個比值」兩種，
     用與 X15 相同的外層交叉驗證與配對 bootstrap，看增益差多少。

八個比值的定義與 X1 相同（成分相加後相除、取對數、在量測年份內置中、跨重複採血平均），
再在分析樣本上做 rank-INT，與主管線一致。
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
SRC, ST = '20260911_v54', '20261030_v65'
RAT = ['R1_PCae_lysoPC_z', 'R2_PCae362_361_z', 'R3_SM_over_PC_z', 'R4_SM240_160_z',
       'R5_lysoPC204_PCaa384_z', 'R6_PCaa384_383_z', 'R7_C16_C0_z', 'R8_C2_C0_z']
LAB = {'R1_PCae_lysoPC_z': 'R1 PC ae / lysoPC', 'R2_PCae362_361_z': 'R2 PC ae C36:2/C36:1',
       'R3_SM_over_PC_z': 'R3 SM / (PC aa + PC ae)', 'R4_SM240_160_z': 'R4 SM C24:0/C16:0',
       'R5_lysoPC204_PCaa384_z': 'R5 lysoPC C20:4 / PC aa C38:4',
       'R6_PCaa384_383_z': 'R6 PC aa C38:4/C38:3', 'R7_C16_C0_z': 'R7 C16/C0',
       'R8_C2_C0_z': 'R8 C2/C0'}
rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))

PHE = pd.read_csv(f'pheno_all_{SRC}.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
C = pd.read_csv(f'{UP}/CKM_clinical_for_microbiome_20260819_V6.csv', low_memory=False)
C['subject_id'] = C.subject_id.astype(str)
C = C.sort_values('visit_no').drop_duplicates('subject_id').copy()
from stage_v10 import master_stage
C['CKM5'] = master_stage(C, 'X21 clinical', quiet=True)
cl = C.set_index('subject_id')
SM_ = [s for s in PHE.subject_id if s in cl.index]

A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
A['CKM5'] = master_stage(A, 'X21 MB825', quiet=True)
mb = A.set_index('subject_id'); S825 = list(A.subject_id)
S743 = [s for s in S825 if s in set(PHE.subject_id)]
ph = PHE.set_index('subject_id')


def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))


def block_test(y, R, Cv):
    """八個比值當一個區塊的概似比檢定（logistic），並回傳每一個的係數。"""
    X1 = np.column_stack([R, Cv]); X0 = Cv
    ok = np.isfinite(X1).all(1) & np.isfinite(y)
    if ok.sum() < 60 or y[ok].sum() < 15: return None
    m1 = sm.GLM(y[ok], sm.add_constant(X1[ok], has_constant='add'),
                family=sm.families.Binomial()).fit()
    m0 = sm.GLM(y[ok], sm.add_constant(X0[ok], has_constant='add'),
                family=sm.families.Binomial()).fit()
    lr = 2 * (m1.llf - m0.llf)
    p = stats.chi2.sf(lr, R.shape[1])
    return dict(lr_stat=lr, lr_df=R.shape[1], lr_p=p, n=int(ok.sum()),
                n_events=int(y[ok].sum()),
                betas=m1.params[1:1 + R.shape[1]].tolist(),
                ses=m1.bse[1:1 + R.shape[1]].tolist(),
                ps=m1.pvalues[1:1 + R.shape[1]].tolist())


# ====================================================== A. 各分期切點
rows, perrat = [], []
for setname, subj, tab in [('Metabolome with a clinical visit', SM_, cl),
                           ('MB743 (microbiota sample)', S743, mb)]:
    R = np.column_stack([rint(ph[r].reindex(subj).values) for r in RAT])
    Cv = np.column_stack([pd.to_numeric(tab.age, errors='coerce').reindex(subj).values,
                          pd.to_numeric(tab.sex, errors='coerce').reindex(subj).values,
                          pd.to_numeric(tab.BMI, errors='coerce').reindex(subj).values])
    stage = pd.to_numeric(tab.CKM5, errors='coerce').reindex(subj).values
    for cut in [1, 2, 3, 4]:
        y = np.where(np.isnan(stage), np.nan, (stage >= cut).astype(float))   # 20261107：無分期者不得當成非事件
        res = block_test(y, R, Cv)
        if res is None: continue
        rows.append(dict(sample=setname, boundary=f'stage >= {cut}', n=res['n'],
                         n_events=res['n_events'], lr_stat=res['lr_stat'],
                         lr_df=res['lr_df'], lr_p=res['lr_p']))
        for j, r_ in enumerate(RAT):
            perrat.append(dict(sample=setname, boundary=f'stage >= {cut}', ratio=LAB[r_],
                               beta=res['betas'][j], se=res['ses'][j], p=res['ps'][j],
                               n=res['n']))
BL = pd.DataFrame(rows); PR = pd.DataFrame(perrat)
# 每個樣本 × 每個切點內，對八個比值做 BH
PR['q'] = np.nan
for key, g in PR.groupby(['sample', 'boundary']):
    pv = g.p.values; o = np.argsort(pv); q = np.empty(len(pv)); run = 1.0
    for i, k in enumerate(o[::-1]):
        run = min(run, pv[k] * len(pv) / (len(pv) - i)); q[k] = run
    PR.loc[g.index, 'q'] = q
BL.to_csv(f'TableSp_eight_ratio_block_{ST}.csv', index=False)
PR.to_csv(f'TableSo_eight_ratio_per_ratio_{ST}.csv', index=False)
pd.set_option('display.width', 220)
print('── A. 八個比值當一個區塊，對每一個分期切點 ──')
print(BL.round(4).to_string(index=False))
print('\n各比值（BH 在同一個切點內的八個之間）')
print(PR[PR.q < 0.10].round(4).to_string(index=False) if (PR.q < 0.10).any()
      else '  沒有任何比值在任何切點通過 q < 0.10')
print()

# ====================================================== B. 判別力
AGE = pd.to_numeric(mb.age, errors='coerce'); SEX = pd.to_numeric(mb.sex, errors='coerce')
BMI = pd.to_numeric(mb.BMI, errors='coerce')
Rall = pd.DataFrame({r: rint(ph[r].reindex(S743).values) for r in RAT}, index=S743)
stage743 = pd.to_numeric(mb.CKM5, errors='coerce').reindex(S743).values
END = {'CKM stage 3 or 4': (stage743 >= 3).astype(float),
       'CKM stage 2 or higher': (stage743 >= 2).astype(float),
       'Diabetes': pd.to_numeric(mb.C_DM, errors='coerce').reindex(S743).values,
       'Elevated triglycerides': pd.to_numeric(mb.C_highTG, errors='coerce').reindex(S743).values}
BASE = np.column_stack([AGE.reindex(S743).values, SEX.reindex(S743).values,
                        BMI.reindex(S743).values])
MODELS = {'Age, sex, BMI': None, '+ R1 only': ['R1_PCae_lysoPC_z'], '+ all eight ratios': RAT}


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]
    if not y.sum() or y.sum() == len(y): return np.nan
    pos, neg = s[y == 1], s[y == 0]
    r = stats.rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cv_auc(y, cols, seed):
    X = BASE if cols is None else np.column_stack([BASE, Rall[cols].values])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    out = np.full(len(y), np.nan); idx = np.where(ok)[0]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(
            np.zeros(ok.sum()), y[ok]):
        try:
            m = sm.GLM(y[idx[tr]], sm.add_constant(X[idx[tr]], has_constant='add'),
                       family=sm.families.Binomial()).fit()
            out[idx[te]] = m.predict(sm.add_constant(X[idx[te]], has_constant='add'))
        except Exception:
            pass
    return auc(y, out), out


B = 500
rng = np.random.default_rng(20261030)
drows = []
for en, y in END.items():
    point = {mn: cv_auc(y, cols, 20261030)[0] for mn, cols in MODELS.items()}
    boot = {mn: [] for mn in MODELS}
    for b in range(B):
        take = rng.integers(0, len(S743), len(S743))
        yb = y[take]
        if yb.sum() < 20: continue
        Xb = {mn: (BASE[take] if cols is None
                   else np.column_stack([BASE[take], Rall[cols].values[take]]))
              for mn, cols in MODELS.items()}
        for mn in MODELS:
            X = Xb[mn]; ok = np.isfinite(X).all(1) & np.isfinite(yb)
            o = np.full(len(yb), np.nan); idx = np.where(ok)[0]
            try:
                for tr, te in StratifiedKFold(5, shuffle=True,
                                              random_state=20261030 + b).split(
                        np.zeros(ok.sum()), yb[ok]):
                    m = sm.GLM(yb[idx[tr]], sm.add_constant(X[idx[tr]], has_constant='add'),
                               family=sm.families.Binomial()).fit()
                    o[idx[te]] = m.predict(sm.add_constant(X[idx[te]], has_constant='add'))
                boot[mn].append(auc(yb, o))
            except Exception:
                boot[mn].append(np.nan)
    base_v = np.array(boot['Age, sex, BMI'], float)
    for mn in MODELS:
        if mn == 'Age, sex, BMI':
            drows.append(dict(endpoint=en, model=mn, auc=point[mn], delta=0.0,
                              lo=np.nan, hi=np.nan, n_events=int(np.nansum(y))))
            continue
        v = np.array(boot[mn], float); n_ = min(len(v), len(base_v))
        d = v[:n_] - base_v[:n_]; d = d[np.isfinite(d)]
        drows.append(dict(endpoint=en, model=mn, auc=point[mn],
                          delta=point[mn] - point['Age, sex, BMI'],
                          lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                          n_events=int(np.nansum(y))))
DD = pd.DataFrame(drows)
DD.to_csv(f'TableSn_eight_ratio_auc_{ST}.csv', index=False)
print('── B. 判別力：單一 R1 對上八個比值（n = 743，配對 bootstrap 的 ΔAUC）──')
print(DD.round(4).to_string(index=False))
print(f'\n寫出 TableSp / TableSo / TableSn（_{ST}.csv）')
