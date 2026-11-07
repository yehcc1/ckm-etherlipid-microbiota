# -*- coding: utf-8 -*-
"""菌相碰得到的代謝體軸，是不是帶疾病訊號的那一條      20261030_v65

使用者提的架構是：菌相 → 代謝體 → 疾病，而菌相本身不直接到疾病。
這個架構能不能成立，關鍵不是菌相和疾病有沒有關聯，而是——
**菌相推動的那條代謝體軸，和帶疾病訊號的那條代謝體軸，是不是同一條。**

做法：
  1. 在不含 MB743 的 2,936 人上，用八個比值配適各個分期切點的 logistic，
     取八個比值的線性預測子當作「該切點的代謝體疾病軸」。訓練集與測試集完全不重疊，
     所以套到 MB743 時不是自己預測自己。
  2. 先驗證這條軸在 MB743 上真的和該切點有關。
  3. 再問菌相和這條軸有沒有關聯。

八個比值的 rank-INT 在各自樣本內做，共變數與主管線相同（年齡、性別、BMI、定序批次）。
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



UPX = '/mnt/user-data/uploads/CKM+microbiome'
RAT = ['R1_PCae_lysoPC_z', 'R2_PCae362_361_z', 'R3_SM_over_PC_z', 'R4_SM240_160_z',
       'R5_lysoPC204_PCaa384_z', 'R6_PCaa384_383_z', 'R7_C16_C0_z', 'R8_C2_C0_z']
ph = PHE.set_index('subject_id')
Ccl = pd.read_csv(f'{UPX}/CKM_clinical_for_microbiome_20260819_V6.csv', low_memory=False)
Ccl['subject_id'] = Ccl.subject_id.astype(str)
Ccl = Ccl.sort_values('visit_no').drop_duplicates('subject_id')
Ccl['CKM5'] = master_stage(Ccl, 'X22 clinical', quiet=True)
cl = Ccl.set_index('subject_id')
SM_ = [s for s in PHE.subject_id if s in cl.index]
TRAIN = [s for s in SM_ if s not in set(S743)]
print(f'訓練集 {len(TRAIN)} 人；測試集 MB743 {len(S743)} 人；兩者不重疊 '
      f'{len(set(TRAIN) & set(S743)) == 0}')

Rmat = lambda subj: np.column_stack([rint(ph[r].reindex(subj).values) for r in RAT])
rows, wrows = [], []
for cut in [2, 3, 4]:
    _st = pd.to_numeric(cl.CKM5, errors='coerce').reindex(TRAIN).values
    ytr = np.where(np.isnan(_st), np.nan, (_st >= cut).astype(float))   # 20261107：無分期者不得當成非事件
    Xtr = np.column_stack([Rmat(TRAIN),
                           pd.to_numeric(cl.age, errors='coerce').reindex(TRAIN).values,
                           pd.to_numeric(cl.sex, errors='coerce').reindex(TRAIN).values,
                           pd.to_numeric(cl.BMI, errors='coerce').reindex(TRAIN).values])
    ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    m = sm.GLM(ytr[ok], sm.add_constant(Xtr[ok], has_constant='add'),
               family=sm.families.Binomial()).fit()
    w = m.params[1:1 + len(RAT)]
    for k, v in zip(RAT, w):
        wrows.append(dict(boundary=f'stage >= {cut}', ratio=k, weight=float(v)))
    sc = pd.Series(rint((Rmat(S743) @ w)), index=S743)
    yv = pd.Series((pd.to_numeric(mb.CKM5, errors='coerce').reindex(S743).values >= cut
                    ).astype(float), index=S743)
    b0, s0, p0, n0 = ols(yv, [sc] + COVL, S743, 'bin')
    rows.append(dict(boundary=f'stage >= {cut}', test='axis validates in MB743',
                     beta=b0, se=s0, p=p0, n=n0, n_train=int(ok.sum())))
    for xn, xs in EXPO.items():
        b, se, p, n = ols(sc, [xs] + COVL, S743)
        rows.append(dict(boundary=f'stage >= {cut}', test=f'microbiota ({xn}) -> axis',
                         beta=b, se=se, p=p, n=n, n_train=int(ok.sum())))
AX = pd.DataFrame(rows); WT = pd.DataFrame(wrows)
AX.to_csv(f'TableSl_metabolite_disease_axis_{ST}.csv', index=False)
WT.to_csv(f'TableSk_axis_weights_{ST}.csv', index=False)
pd.set_option('display.width', 200)
print(AX.round(4).to_string(index=False)); print()
print(WT.pivot(index='ratio', columns='boundary', values='weight').round(3).to_string())
print(f'\n寫出 TableSl_metabolite_disease_axis_{ST}.csv、TableSk_axis_weights_{ST}.csv')
