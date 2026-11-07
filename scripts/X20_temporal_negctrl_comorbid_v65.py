# -*- coding: utf-8 -*-
"""時間先後、負控制，以及「對 CKM 的效果是不是共病造成的」      20261030_v65

三件事，分開做。

A. 時間先後
   逐次（未平均）的代謝物檔是 04_data_matrices/p180_all.csv，6,704 列、3,688 人，
   有 p180_date。R1 的定義照 X1：各成分相加後相除、取自然對數、在量測年份內置中。
   把每一次訪次對到各人的糞便採檢日之後，可行性如下（腳本會自己印出來）：
   順向設計（採檢 → 之後的 R1）人數太少；絕大多數代謝體訪次落在採檢日**之前**。
   所以能做的是：菌相與「這段期間 R1 的變化」有沒有關係——以採檢當時的 R1 為結局，
   校正 180 天以前那一次的 R1。這不等於時間先後，但可以排除「菌相只是對應到一個
   長年不變的脂質體質」。另外也直接測：菌相與多年前的 R1 有沒有關係。

B. 負控制
   NCO 負控制結局：挑與這條路無關、但共用同一套混淆結構的結局。
   NCE 負控制暴露：把 R1 打亂之後用完全相同的交叉驗證流程重新選菌屬、重建分數，
   再去測真正的 R1。這給出「這條管線自己能製造出多強的關聯」的虛無分布，
   是對整個選擇程序最直接的負控制。

C. 使用者的論點
   菌相對 CKM 分期的效果，是不是只是經由糖尿病與高三酸甘油酯這兩個共病？
   直接檢定：在模型裡放進 TG 與糖尿病，看菌相的係數還剩多少。
   ★ 但 CKM 第二期的判準本身就含高三酸甘油酯與血糖異常，所以對五級分期做這件事
   等於把結局的一部分當共變數調掉。沒有這個循環的終點是「第三或第四期」
   （由心臟、腎臟標記與心血管診斷判定），所以兩個終點都報。
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import KFold
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'
SRC, ST = '20260911_v54', '20261030_v65'
N_PERM = 200

rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))

# ----------------------------------------------------------------- 基礎資料
PHE = pd.read_csv(f'pheno_all_{SRC}.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
from stage_v10 import master_stage
A['CKM5'] = master_stage(A, 'X20', quiet=True)
A['collection_date'] = pd.to_datetime(A.collection_date, errors='coerce')
mb = A.set_index('subject_id'); S825 = list(A.subject_id)
S743 = [s for s in S825 if s in set(PHE.subject_id)]
PAN = pd.read_csv(f'panel_scores_{SRC}.csv', index_col=0); PAN.index = PAN.index.astype(str)
R1 = pd.Series(rint(PHE.set_index('subject_id').R1_PCae_lysoPC_z.values),
               index=PHE.subject_id.astype(str))
age = pd.to_numeric(mb.age, errors='coerce'); sex = pd.to_numeric(mb.sex, errors='coerce')
bmi = pd.to_numeric(mb.BMI, errors='coerce')
# Batch12 只有 1 個人。單人批次會讓那一列的槓桿值等於 1，HC3 的分母 (1-h)^2 變成 0，
# 標準誤就會變成 inf、p 變成 1 —— 不是降秩，是 HC3 對單人群組無定義。
# 把人數少於 5 的批次併成一個 other。
_bc = mb.batch.value_counts()
_small = set(_bc[_bc < 5].index)
_batch = mb.batch.where(~mb.batch.isin(_small), 'BatchOther')
BAT = pd.get_dummies(_batch, drop_first=True).astype(float)
print(f'批次：{mb.batch.nunique()} 個，其中 {len(_small)} 個人數 < 5 已併入 BatchOther')
COV = pd.concat([pd.DataFrame({'age': age, 'sex': sex, 'bmi': bmi}), BAT], axis=1)
EXPO = {'Pre-specified score': PAN.score_prior, 'CV panel': PAN.panel_oof}


def drop_const(X):
    """先去掉常數欄，再用帶樞軸的 QR 去掉線性相依欄。

    批次 dummy 在某些子樣本裡會彼此相依（Batch12 只有 1 人），設計矩陣一降秩，
    HC3 的標準誤就會變成 inf，p 值變成 1。這裡保證交出去的是滿秩矩陣，
    而且永遠保留最前面的暴露欄（QR 樞軸會優先挑它們，因為它們方差最大之外，
    下面也明確把前 k 欄固定保留）。"""
    from scipy.linalg import qr
    keep = np.ones(X.shape[1], bool)
    for j in range(1, X.shape[1]):
        if np.nanstd(X[:, j]) < 1e-12: keep[j] = False
    Xk = X[:, keep]
    idx = np.where(keep)[0]
    _, _, piv = qr(Xk, mode='economic', pivoting=True)
    rank = np.linalg.matrix_rank(Xk)
    sel = sorted(piv[:rank])
    # 暴露欄（原本的第 0 欄）一定要留著，否則係數就不是我們要的那個
    if 0 not in sel: sel = [0] + [c for c in sel if c != 0]
    out = np.ones(X.shape[1], bool) & False
    out[idx[sel]] = True
    return X[:, out]


def hc3(X, y):
    """與 X2 完全相同的 HC3：槓桿值截在 0.999。

    世代裡有單人批次，那一列的槓桿值正好是 1，未截斷的 HC3 分母是 0，
    標準誤會變成 inf。主管線本來就是這樣處理的，這裡沿用同一個估計式，
    不另外發明一套。"""
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))


def ols(y, cols, idx, kind='lin'):
    X = np.column_stack([np.asarray(c.reindex(idx).values, float) if hasattr(c, 'reindex')
                         else np.asarray(c, float) for c in cols])
    yv = np.asarray(y.reindex(idx).values, float) if hasattr(y, 'reindex') else np.asarray(y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(yv)
    if ok.sum() < 40: return np.nan, np.nan, np.nan, 0
    Xd = sm.add_constant(drop_const(X[ok]), has_constant='add')
    if kind == 'bin':
        m = sm.GLM(yv[ok], Xd, family=sm.families.Binomial()).fit(cov_type='HC3')
        return m.params[1], m.bse[1], m.pvalues[1], int(ok.sum())
    b, se = hc3(Xd, yv[ok])
    p = 2 * stats.t.sf(abs(b[1] / se[1]), ok.sum() - Xd.shape[1])
    return b[1], se[1], p, int(ok.sum())


COVL = [COV[c] for c in COV.columns]

# ======================================================= A. 時間先後
D = pd.read_csv(f'{UP}/analysis/04_data_matrices/p180_all.csv', low_memory=False)
D['sid'] = D.subject_base.astype(str)
D['p180_date'] = pd.to_datetime(D.p180_date, errors='coerce')
cols = list(D.columns)
PCae = [c for c in cols if re.match(r'^PC ae ', c)]
LYSO = [c for c in cols if re.match(r'^lysoPC ', c)]
nm = D[PCae].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
dn = D[LYSO].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
r = nm / dn
lr = np.log(r.where(r > 0))
year = pd.to_numeric(D.test_year, errors='coerce')
V = pd.DataFrame({'sid': D.sid, 'date': D.p180_date,
                  'R1v': (lr - lr.groupby(year).transform('mean')).values})
V = V[V.sid.isin(set(S743))].dropna(subset=['date', 'R1v']).copy()
V['dt'] = (V.date - V.sid.map(mb.collection_date)).dt.days

FEAS = []
for lab, cond in [('visits at or before the stool sample (dt <= 30 d)', V.dt <= 30),
                  ('visits after the stool sample (dt > 90 d)', V.dt > 90),
                  ('visits more than 180 d before the stool sample', V.dt < -180),
                  ('visits more than 365 d before the stool sample', V.dt < -365)]:
    FEAS.append(dict(window=lab, participants=int(V[cond].sid.nunique()),
                     visits=int(cond.sum())))
pre = set(V[V.dt <= 30].sid); post = set(V[V.dt > 90].sid)
FEAS.append(dict(window='both before and after (forward design)',
                 participants=len(pre & post), visits=np.nan))
FE = pd.DataFrame(FEAS)
print('── A. 時間先後的可行性 ──'); print(FE.to_string(index=False)); print()

now = V[V.dt <= 30].sort_values('dt').groupby('sid').last()        # 離採檢最近且不晚於採檢
past = V[V.dt < -180].sort_values('dt').groupby('sid').first()     # 最早的一次（>180 天前）
LAG = pd.DataFrame({'R1_now': now.R1v, 'R1_past': past.R1v,
                    'gap_days': (now.date - past.date).dt.days}).dropna()
print(f'可做「校正多年前 R1」的人數：{len(LAG)}；'
      f'間隔中位數 {int(LAG.gap_days.median())} 天（四分位 '
      f'{int(LAG.gap_days.quantile(.25))}–{int(LAG.gap_days.quantile(.75))}）\n')
idx = list(LAG.index)
rows = []
for xn, xs in EXPO.items():
    b, se, p, n = ols(pd.Series(rint(LAG.R1_now.values), index=idx), [xs] + COVL, idx)
    rows.append(dict(model='R1 at the stool visit (no lag adjustment)', exposure=xn,
                     beta=b, se=se, p=p, n=n))
    b, se, p, n = ols(pd.Series(rint(LAG.R1_past.values), index=idx), [xs] + COVL, idx)
    rows.append(dict(model='R1 measured >180 d earlier', exposure=xn, beta=b, se=se, p=p, n=n))
    b, se, p, n = ols(pd.Series(rint(LAG.R1_now.values), index=idx),
                      [xs, pd.Series(rint(LAG.R1_past.values), index=idx)] + COVL, idx)
    rows.append(dict(model='R1 at the stool visit, adjusted for the earlier R1',
                     exposure=xn, beta=b, se=se, p=p, n=n))
LAGR = pd.DataFrame(rows)
LAGR.to_csv(f'TableSu_temporal_{ST}.csv', index=False)
print(LAGR.round(4).to_string(index=False)); print()

# ======================================================= B. 負控制
# B1 負控制結局
NCO = {
    'R6 PC aa C38:4/C38:3 (FADS desaturation index)': ('lin', 'R6_PCaa384_383_z'),
    'R4 SM C24:0/C16:0': ('lin', 'R4_SM240_160_z'),
    'R5 lysoPC a C20:4 / PC aa C38:4': ('lin', 'R5_lysoPC204_PCaa384_z'),
}
ph = PHE.set_index('subject_id')
LABNCO = {'Serum creatinine': 'lab_creatinine', 'Haemoglobin': 'lab_Hemoglobin',
          'Platelet count': 'lab_Platelet', 'Serum sodium': 'lab_Sodium',
          'TSH': 'lab_TSH'}
rows = []
for xn, xs in EXPO.items():
    b, se, p, n = ols(R1, [xs] + COVL, S743)
    rows.append(dict(outcome='R1 PC ae / lysoPC (the primary outcome)', kind='primary',
                     exposure=xn, beta=b, se=se, p=p, n=n))
    for lab, (k, col) in NCO.items():
        y = pd.Series(rint(ph[col].reindex(S743).values), index=S743)
        b, se, p, n = ols(y, [xs] + COVL, S743)
        rows.append(dict(outcome=lab, kind='negative control (metabolite)',
                         exposure=xn, beta=b, se=se, p=p, n=n))
    for lab, col in LABNCO.items():
        y = pd.Series(rint(pd.to_numeric(mb[col], errors='coerce').reindex(S825).values),
                      index=S825)
        b, se, p, n = ols(y, [xs] + COVL, S825)
        rows.append(dict(outcome=lab, kind='negative control (clinical)',
                         exposure=xn, beta=b, se=se, p=p, n=n))
NC = pd.DataFrame(rows)
# 負控制之間做 BH：一次看八個結局，不校正會被偶然的 P<0.05 誤導
for xn in EXPO:
    m_ = (NC.exposure == xn) & (NC.kind != 'primary')
    pv = NC.loc[m_, 'p'].values
    o = np.argsort(pv); q = np.empty(len(pv)); run = 1.0
    for i, k in enumerate(o[::-1]):
        run = min(run, pv[k] * len(pv) / (len(pv) - i)); q[k] = run
    NC.loc[m_, 'q_across_negative_controls'] = q
NC.to_csv(f'TableSt_negative_controls_{ST}.csv', index=False)
print('── B1. 負控制結局 ──')
print(NC[['outcome','kind','exposure','beta','se','p','q_across_negative_controls','n']].round(4).to_string(index=False)); print()

# B2 負控制暴露：把 R1 打亂後用同一條流程重建分數
GC = pd.read_csv(f'{HUB}/abundance_genus_counts.csv', index_col=0)
GX = GC.loc[A.sample_id]; KEEP = ((GX > 0).mean(0) >= 0.10)
GXk = GX.loc[:, KEEP[KEEP].index]
CL = np.log(GXk + 0.5); CL = CL.sub(CL.mean(1), axis=0); CL.index = A.subject_id.values
from xnames import labels as taxa_labels
CL.columns = taxa_labels(list(GXk.columns))
Z = (CL - CL.mean()) / CL.std()
Zi = Z.reindex(S743)
Cv = COV.reindex(S743).values.astype(float)
Cv = drop_const(np.column_stack([np.ones(len(S743)), Cv]))
y_true = R1.reindex(S743).values.astype(float)


def build_panel(yv):
    """與 X2 相同的規則：5 fold、每 fold 內 P<0.01、sign 權重、out-of-fold 計分。"""
    out = np.full(len(S743), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=20260822).split(np.arange(len(S743))):
        w = np.zeros(Z.shape[1])
        Xtr = Zi.values[tr]; ytr = yv[tr]; Ctr = Cv[tr]
        for j in range(Z.shape[1]):
            X = np.column_stack([Xtr[:, j], Ctr])
            ok = np.isfinite(X).all(1) & np.isfinite(ytr)
            if ok.sum() < 30: continue
            Xi = np.linalg.pinv(X[ok].T @ X[ok]); b = Xi @ (X[ok].T @ ytr[ok])
            e = ytr[ok] - X[ok] @ b
            se = np.sqrt((e @ e / (ok.sum() - X.shape[1])) * Xi[0, 0])
            if 2 * stats.t.sf(abs(b[0] / se), ok.sum() - X.shape[1]) < 0.01:
                w[j] = np.sign(b[0])
        out[te] = Zi.values[te] @ w
    s = pd.Series(out, index=S743)
    return (s - s.mean()) / s.std()


obs_b, obs_se, obs_p, obs_n = ols(R1, [build_panel(y_true)] + COVL, S743)
rng = np.random.default_rng(20261030)
null_b = []
for i in range(N_PERM):
    yp = y_true[rng.permutation(len(y_true))]
    b, _, _, _ = ols(R1, [build_panel(yp)] + COVL, S743)
    if np.isfinite(b): null_b.append(abs(b))
    if (i + 1) % 50 == 0: print(f'   置換 {i + 1}/{N_PERM}')
null_b = np.array(null_b)
np.savez(f'perm_null_{ST}.npz', null_abs_beta=null_b, observed_beta=obs_b, observed_p=obs_p)
perm_p = float((null_b >= abs(obs_b)).mean())
PERM = pd.DataFrame([dict(test='CV panel rebuilt on permuted R1, tested against the true R1',
                          observed_beta=obs_b, observed_p=obs_p, n=obs_n,
                          n_permutations=len(null_b),
                          null_abs_beta_median=float(np.median(null_b)),
                          null_abs_beta_p95=float(np.percentile(null_b, 95)),
                          permutation_p=perm_p)])
PERM.to_csv(f'TableSs_permutation_nce_{ST}.csv', index=False)
print('\n── B2. 負控制暴露（置換 R1 之後重建分數）──')
print(PERM.round(5).to_string(index=False)); print()

# ======================================================= C. 共病調整
tg = pd.Series(rint(np.log(pd.to_numeric(mb.lab_triglyceride, errors='coerce').values)),
               index=mb.index)
dm = pd.to_numeric(mb.C_DM, errors='coerce')
ck5 = pd.Series(rint(pd.to_numeric(mb.CKM5, errors='coerce').values), index=mb.index)
ck34 = (pd.to_numeric(mb.CKM5, errors='coerce') >= 3).astype(float)
rows = []
for en, y, kind, circ in [('CKM stage (5-level)', ck5, 'lin', 'contains TG and glycaemia criteria'),
                          ('CKM stage 3 or 4', ck34, 'bin', 'none')]:
    for xn, xs in EXPO.items():
        for mn, extra in [('exposure only', []),
                          ('+ ether-lipid ratio', [R1]),
                          ('+ triglycerides and diabetes', [tg, dm]),
                          ('+ ether-lipid ratio, triglycerides and diabetes', [R1, tg, dm])]:
            b, se, p, n = ols(y, [xs] + extra + COVL, S743, kind)
            rows.append(dict(outcome=en, circularity=circ, exposure=xn, model=mn,
                             beta=b, se=se, p=p, n=n))
    # R1 自己在同一組模型裡
    for mn, extra in [('exposure only', []), ('+ triglycerides and diabetes', [tg, dm])]:
        b, se, p, n = ols(y, [R1] + extra + COVL, S743, kind)
        rows.append(dict(outcome=en, circularity=circ, exposure='Ether-lipid ratio',
                         model=mn, beta=b, se=se, p=p, n=n))
CO = pd.DataFrame(rows)
CO.to_csv(f'TableSr_comorbidity_{ST}.csv', index=False)
print('── C. 放進糖尿病與三酸甘油酯之後 ──')
print(CO.round(4).to_string(index=False))
print(f'\n寫出 TableSu_temporal / TableSt_negative_controls / '
      f'TableSs_permutation_nce / TableSr_comorbidity（都是 _{ST}.csv）')
