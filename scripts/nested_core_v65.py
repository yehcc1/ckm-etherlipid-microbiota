# -*- coding: utf-8 -*-
"""nested CV 的資料與函式，供 X15 與 X17 共用      20261030_v65

原本這些程式碼寫在 X15 裡，X17 要重畫 Figure 5 時需要同一套前處理與選擇規則。
為了不讓兩支腳本各算一次（這份稿子已經被這類重複算兩次坑過三回），抽成模組。

以下是原本的說明：


外審 issue 08 說得對。原本的做法是：

  第一層  用全部 743 人、依 R1 選菌屬，以 5-fold 產生 out-of-fold 的 panel 分數
  第二層  把那個分數當成一個固定欄位，另外用一組 fold 跑臨床結局的 CV

每個人自己的 R1 確實沒有進到自己的分數裡，但「選哪些菌屬」用到了全部 743 人的
R1，而第二層的外層測試集是另一組切分，所以外層測試集的人會經由其他外層測試集
的人影響到自己的分數。這是橫向洩漏。bootstrap 也只對 out-of-fold 的預測重抽，
沒有把選擇變異包進去，信賴區間偏窄。

這支腳本做兩件事：

  A  真正的 nested CV：外層切分固定之後，前處理（CLR 標準化）、依 R1 選菌屬、
     panel 分數的標準化、模型配適全部只在外層訓練集內完成，再預測外層測試集。
     信賴區間用整條 pipeline 的 bootstrap，不是對預測重抽。

  B  82 人保留集：沒有代謝體的 82 人從頭到尾沒有 R1，不可能進入任何一次選擇，
     是乾淨的外部樣本。用 743 人建模，預測這 82 人。事件數很少，CI 會很寬，
     這點直接寫出來。

兩種 panel 都跑：P<0.01 sign-weight（正文所稱的 microbiota panel）與 top-k
（Figure 5A 的面板大小曲線所用）。top-k 的 k 在外層訓練集內用內層 CV 挑，
不看外層測試集。
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, KFold
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'
ST = '20261030_v65'
B_BOOT = 500                       # 整條 pipeline 的 bootstrap 次數
SEED_OUTER, SEED_INNER = 20261030, 20261031

rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))

# ------------------------------------------------------------------ 資料
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
from stage_v10 import master_stage
A['CKM5'] = master_stage(A, 'X15 MB825', quiet=True)
S825 = list(A.subject_id); mb = A.set_index('subject_id')
S743 = [s for s in S825 if s in set(PHE.subject_id)]
S82 = [s for s in S825 if s not in set(S743)]
assert len(S743) == 743 and len(S82) == 82, (len(S743), len(S82))

GC = pd.read_csv(f'{HUB}/abundance_genus_counts.csv', index_col=0)
GX = GC.loc[A.sample_id]
KEEPG = ((GX > 0).mean(0) >= 0.10)
GXk = GX.loc[:, KEEPG[KEEPG].index]
from xnames import labels as taxa_labels
RAWCLR = np.log(GXk + 0.5); RAWCLR = RAWCLR.sub(RAWCLR.mean(1), axis=0)
RAWCLR.index = A.subject_id.values
RAWCLR.columns = taxa_labels(list(GXk.columns))      # CLR，尚未標準化

ph = PHE.set_index('subject_id')
R1 = pd.Series(rint(ph.R1_PCae_lysoPC_z.values), index=ph.index)     # rank-INT over 3,688
age = pd.to_numeric(mb.age, errors='coerce')
sex = pd.to_numeric(mb.sex, errors='coerce')
bmi = pd.to_numeric(mb.BMI, errors='coerce')
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float)

END = {'Diabetes': pd.to_numeric(mb.C_DM, errors='coerce'),
       'CKM stage 3 or 4': (pd.to_numeric(mb.CKM5, errors='coerce') >= 3).astype(float),
       'Elevated triglycerides': pd.to_numeric(mb.C_highTG, errors='coerce')}


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]
    if not y.sum() or y.sum() == len(y): return np.nan
    pos, neg = s[y == 1], s[y == 0]
    r = stats.rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ------------------------------------------------------- 只用訓練列的選擇
def select_stats(train, Zc):
    """train 內每一屬對 R1 的校正關聯；共變數含批次，p 用 t 分布。

    Zc 是只用訓練列的平均數與標準差標準化之後的 CLR 表。"""
    tr_r1 = [s for s in train if s in R1.index and np.isfinite(R1.get(s, np.nan))]
    if len(tr_r1) < 60:
        n_ = Zc.shape[1]
        return np.full(n_, np.nan), np.full(n_, np.nan), np.full(n_, np.nan)
    Cv = np.column_stack([np.ones(len(tr_r1)), age.reindex(tr_r1).values,
                          sex.reindex(tr_r1).values, bmi.reindex(tr_r1).values,
                          BAT.reindex(tr_r1).values])
    yv = R1.reindex(tr_r1).values
    nb = Zc.shape[1]; beta = np.full(nb, np.nan); tt = np.full(nb, np.nan)
    Zt = Zc.reindex(tr_r1).values
    for j in range(nb):
        X = np.column_stack([Zt[:, j], Cv])
        ok = np.isfinite(X).all(1) & np.isfinite(yv)
        if ok.sum() < 30: continue
        Xi = np.linalg.pinv(X[ok].T @ X[ok]); b = Xi @ (X[ok].T @ yv[ok])
        e = yv[ok] - X[ok] @ b
        se = np.sqrt((e @ e / (ok.sum() - X.shape[1])) * Xi[0, 0])
        beta[j] = b[0]; tt[j] = b[0] / se
    df = max(len(tr_r1) - 5 - BAT.shape[1], 1)
    p = 2 * stats.t.sf(np.abs(tt), df)
    return beta, p, tt


def panel_weights(beta, p, tt, rule, k=None):
    w = np.zeros(len(beta))
    if rule == 'p01':
        hit = np.isfinite(p) & (p < 0.01)
        w[hit] = np.sign(beta[hit])
    else:
        a = np.where(np.isfinite(tt), np.abs(tt), 0)
        top = np.argsort(-a)[:k]
        w[top] = np.sign(beta[top])
    return w


def fit_predict(train, test, cols, score_tr, score_te, y):
    """在訓練列上配適 logistic，預測測試列。score_* 是已經算好的 panel 欄位。"""
    def design(idx, sc):
        parts = []
        for c in cols:
            if c == 'panel': parts.append(sc)
            elif c == 'r1': parts.append(R1.reindex(idx).values)
            elif c == 'age': parts.append(age.reindex(idx).values)
            elif c == 'sex': parts.append(sex.reindex(idx).values)
            elif c == 'bmi': parts.append(bmi.reindex(idx).values)
        return np.column_stack(parts)
    Xtr, Xte = design(train, score_tr), design(test, score_te)
    ytr = y.reindex(train).values.astype(float)
    ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    if ok.sum() < 40 or ytr[ok].sum() < 5: return np.full(len(test), np.nan)
    try:
        m = sm.GLM(ytr[ok], sm.add_constant(Xtr[ok], has_constant='add'),
                   family=sm.families.Binomial()).fit()
    except Exception:
        return np.full(len(test), np.nan)
    out = np.full(len(test), np.nan)
    ok2 = np.isfinite(Xte).all(1)
    out[ok2] = m.predict(sm.add_constant(Xte[ok2], has_constant='add'))
    return out


MODELS = {'Age, sex, BMI': ['age', 'sex', 'bmi'],
          '+ microbiota panel': ['age', 'sex', 'bmi', 'panel'],
          '+ ether-lipid ratio': ['age', 'sex', 'bmi', 'r1'],
          '+ microbiota panel and ether-lipid ratio': ['age', 'sex', 'bmi', 'panel', 'r1']}
KGRID = [10, 20, 30, 40, 60]


def nested_predictions(subjects, y, rule, seed=SEED_OUTER, k=None):
    """回傳每個人的外層預測；前處理、選擇、標準化全部只在外層訓練集內做。"""
    subj = np.array(subjects)
    yv = y.reindex(subj).values.astype(float)
    ok = np.isfinite(yv)
    preds = {m: np.full(len(subj), np.nan) for m in MODELS}
    kept_k = []
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(
            np.zeros(ok.sum()), yv[ok]):
        idx = np.where(ok)[0]
        train = list(subj[idx[tr_i]]); test = list(subj[idx[te_i]])
        # 前處理：只用訓練列的平均數與標準差
        mu, sd = RAWCLR.reindex(train).mean(), RAWCLR.reindex(train).std()
        Zc = (RAWCLR - mu) / sd
        beta, p, tt = select_stats(train, Zc)
        if rule == 'topk_fixed':
            # k 由呼叫端固定（面板大小曲線用），其餘流程完全相同
            kk = int(k)
        elif rule == 'topk':
            # k 在訓練集內用內層 CV 選，不看外層測試集
            best_k, best_a = KGRID[0], -1
            inner = list(StratifiedKFold(3, shuffle=True, random_state=SEED_INNER).split(
                np.zeros(len(train)), y.reindex(train).values.astype(float)))
            # 每個內層 fold 的統計只算一次，k 只改權重，不改統計
            cache = []
            for itr, ite in inner:
                tr2 = [train[i] for i in itr]; te2 = [train[i] for i in ite]
                mu2, sd2 = RAWCLR.reindex(tr2).mean(), RAWCLR.reindex(tr2).std()
                Z2 = (RAWCLR - mu2) / sd2
                cache.append((itr, ite, Z2.reindex(tr2).values, Z2.reindex(te2).values,
                              *select_stats(tr2, Z2)))
            for k in KGRID:
                sc_in = np.full(len(train), np.nan)
                for itr, ite, Ztr2, Zte2, b2, p2, t2 in cache:
                    w2 = panel_weights(b2, p2, t2, 'topk', k)
                    base2 = Ztr2 @ w2
                    sc_in[ite] = (Zte2 @ w2 - np.nanmean(base2)) / (np.nanstd(base2) or 1)
                a = auc(y.reindex(train).values.astype(float), sc_in)
                if np.isfinite(a) and a > best_a: best_a, best_k = a, k
            kk = best_k
        else:
            kk = None
        kept_k.append(kk)
        w = panel_weights(beta, p, tt, 'topk' if rule.startswith('topk') else rule, kk)
        raw_tr = Zc.reindex(train).values @ w
        m_, s_ = np.nanmean(raw_tr), (np.nanstd(raw_tr) or 1.0)
        sc_tr = (raw_tr - m_) / s_
        sc_te = (Zc.reindex(test).values @ w - m_) / s_
        for mn, cols in MODELS.items():
            preds[mn][idx[te_i]] = fit_predict(train, test, cols, sc_tr, sc_te, y)
    return preds, yv, kept_k


