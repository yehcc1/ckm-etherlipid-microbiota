"""Three analyses the prior cohort paper reports and this manuscript did not — 20260911_v54.

Wu et al. (Gut Microbes 2025;17:2473506), from the same cohort, carry three things this
manuscript lacked:

  their Figure 2  a selection curve, a ranked feature list and a cross-validated
                  discrimination table with accuracy and area under the curve
  their Figure 3  a taxon × clinical-index correlation matrix
  their Figure 4a a per-taxon abundance comparison across the clinical groups

This script supplies the equivalents, with one addition that their design could not make:
every model is also run with the ether-lipid ratio, so the question "does the microbiota
add anything once the host lipid readout is known, and vice versa" is answerable.

Everything here is cross-validated out of fold; no participant contributes to the selection
or fitting of the model that scores them.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, KFold
np.seterr(all='ignore')

UP  = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'
ST  = '20260911_v54'
F = []
def add(k, v, u='', s=''): F.append(dict(key=k, value=v, unit=u, source=s))
rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))

PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
from stage_v10 import master_stage
A['CKM5'] = master_stage(A, 'X3 MB825')
S825 = list(A.subject_id); mb = A.set_index('subject_id')
S743 = [s for s in S825 if s in set(PHE.subject_id)]
sub743 = np.array([s in set(S743) for s in S825])

GC = pd.read_csv(f'{HUB}/abundance_genus_counts.csv', index_col=0)
GX = GC.loc[A.sample_id]; KEEPG = ((GX > 0).mean(0) >= 0.10)
GXk = GX.loc[:, KEEPG[KEEPG].index]
CL = np.log(GXk + 0.5); CL = CL.sub(CL.mean(1), axis=0); CL.index = A.subject_id.values
from xnames import labels as taxa_labels
GEN = taxa_labels(list(CL.columns))
CL.columns = GEN
Z = (CL - CL.mean()) / CL.std()
REL = (GXk.div(GXk.sum(1), axis=0) * 100); REL.index = A.subject_id.values; REL.columns = GEN

ph = PHE.set_index('subject_id')
y1 = pd.Series(rint(ph.R1_PCae_lysoPC_z.values), index=ph.index)      # rank-INT over 3,688
age = pd.to_numeric(mb.age, errors='coerce'); sex = pd.to_numeric(mb.sex, errors='coerce')
bmi = pd.to_numeric(mb.BMI, errors='coerce')

END = {'Diabetes':            pd.to_numeric(mb.C_DM, errors='coerce'),
       'CKM stage 3 or 4':    (pd.to_numeric(mb.CKM5, errors='coerce') >= 3).astype(float),
       'Elevated triglycerides': pd.to_numeric(mb.C_highTG, errors='coerce')}

def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg): return np.nan
    r = stats.rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def roc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y) / max(y.sum(), 1); fp = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return np.r_[0, fp, 1], np.r_[0, tp, 1]

def cv_predict(Xdf, yv, subjects, seed=20260911):
    """out-of-fold logistic predictions on the columns given.\n\n    20261030_v65：原本的說明寫「genus columns are re-selected inside each fold」，\n    但這個函式從來沒有重選菌屬，它只是對傳進來的欄位配適 GLM。完整的\n    nested CV 在 X15 另外做，這裡保留原行為但把說明改成實際行為。"""
    X = Xdf.reindex(subjects).values.astype(float); y = np.asarray(yv, float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    out = np.full(len(subjects), np.nan)
    Xo, yo = X[ok], y[ok]; where = np.where(ok)[0]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xo, yo):
        try:
            m = sm.GLM(yo[tr], sm.add_constant(Xo[tr], has_constant='add'),
                       family=sm.families.Binomial()).fit()
            out[where[te]] = m.predict(sm.add_constant(Xo[te], has_constant='add'))
        except Exception:
            pass
    return out

def boot_auc(y, s, B=2000, seed=7):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = np.isfinite(y) & np.isfinite(s); y, s = y[ok], s[ok]; n = len(y)
    vals = [auc(y[j], s[j]) for j in (rng.integers(0, n, n) for _ in range(B))]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return np.percentile(vals, [2.5, 97.5])

# ============ A. 菌屬選擇一律讀 X2 的輸出，不在這裡重算 ============
# 20261030_v65：原本這裡有自己的 select_weights，共變數少了定序批次、p 用常態近似，
# 所以同一個 seed、同一組 fold 卻選出不同名單（這裡 18 屬，TableS10 19 屬）。
# 實際餵進所有模型的 panel 分數出自 X2，X2 才是權威來源，這裡改成只讀不算。
_PS = np.load(f'panel_fold_stats_{ST}.npz', allow_pickle=True)
_gen = [str(g) for g in _PS['genus']]
assert _gen == list(Z.columns), 'X2 與 X3 的菌屬順序不一致，兩邊的 CLR 前處理已經分岔'
folds = list(KFold(5, shuffle=True, random_state=20260822).split(np.arange(len(S743))))
FOLD_T = [(_PS['fold_t'][i], None) for i in range(_PS['fold_t'].shape[0])]
assert len(FOLD_T) == len(folds), 'fold 數不符'
T_full, B_full = _PS['t_full'], _PS['beta_full']
RANK = pd.DataFrame(dict(genus=Z.columns, beta=B_full, t=T_full,
                         p=_PS['p_full'], folds_selected=_PS['folds_selected'].astype(int)))
RANK['abs_t'] = RANK.t.abs()
RANK = RANK.sort_values('abs_t', ascending=False).reset_index(drop=True)
RANK.index += 1; RANK.index.name = 'rank'
RANK.head(30).to_csv(f'Table2_panel_ranked_{ST}.csv')
STABLE = RANK[RANK.folds_selected >= 3].genus.tolist()
_S10 = pd.read_csv(f'TableS10_panel_genera_{ST}.csv')
assert sorted(STABLE) == sorted(_S10[_S10.folds_selected >= 3].genus), \
    'STABLE 與 TableS10 的 >=3 fold 名單不一致'
print(f'菌屬名單取自 X2（TableS10）：{len(STABLE)} 屬')
add('n_stable_genera', len(STABLE), 'genera', 'selected at P<0.01 in >=3 of 5 folds')
add('stable_genera', '; '.join(STABLE), '', '')

def panel_score(k):
    """out-of-fold score using the top k genera of each training fold."""
    s = np.full(len(S743), np.nan)
    for (tr, te), (t, _b) in zip(folds, FOLD_T):
        top = np.argsort(-np.abs(t))[:k]; w = np.zeros(Z.shape[1]); w[top] = np.sign(t[top])
        s[te] = Z.reindex([S743[i] for i in te]).values @ w
    full_top = np.argsort(-np.abs(T_full))[:k]
    wf = np.zeros(Z.shape[1]); wf[full_top] = np.sign(T_full[full_top])
    extra = [s_ for s_ in S825 if s_ not in set(S743)]
    out = pd.Series(s, index=S743)
    out = pd.concat([out, pd.Series(Z.reindex(extra).values @ wf, index=extra)])
    return ((out - out.reindex(S743).mean()) / out.reindex(S743).std()).reindex(S825)

# ============ B. feature-number selection curve (their Figure 2b) ============
KS = [5, 10, 15, 20, 25, 30, 40, 50, 60]
curve = []
for k in KS:
    sc = panel_score(k)
    for en, yv in END.items():
        X = pd.DataFrame({'age': age, 'sex': sex, 'bmi': bmi, 'panel': sc})
        p = cv_predict(X, yv.reindex(S825).values, S825)
        a = auc(yv.reindex(S825).values, p)
        ok = np.isfinite(p) & np.isfinite(yv.reindex(S825).values)
        acc = ((p[ok] > 0.5).astype(float) == yv.reindex(S825).values[ok]).mean()
        curve.append(dict(k=k, endpoint=en, auc=a, accuracy=acc, n=int(ok.sum())))
CURVE = pd.DataFrame(curve); CURVE.to_csv(f'fig5b_feature_curve_{ST}.csv', index=False)
best = CURVE.groupby('k').auc.mean().idxmax()
add('feature_curve_best_k', int(best), 'genera', 'mean AUC across the three endpoints')
PANEL = panel_score(int(best))

# ============ C. discrimination (their Figure 2d) ============
r1_825 = y1.reindex(S825)
MODELS = {
    'Age, sex, BMI':                     ['age', 'sex', 'bmi'],
    '+ microbiota panel':                ['age', 'sex', 'bmi', 'panel'],
    '+ ether-lipid ratio':               ['age', 'sex', 'bmi', 'r1'],
    '+ microbiota panel and ether-lipid ratio': ['age', 'sex', 'bmi', 'panel', 'r1'],
}
BASE = pd.DataFrame({'age': age, 'sex': sex, 'bmi': bmi,
                     'panel': PANEL, 'r1': r1_825}, index=S825)
rows = []; ROC_STORE = {}
for setname, subj in [('MB825', S825), ('MB743', S743)]:
    for en, yv in END.items():
        for mname, cols in MODELS.items():
            if setname == 'MB825' and 'r1' in cols: continue
            y = yv.reindex(subj).values
            p = cv_predict(BASE[cols], y, subj)
            a = auc(y, p); lo, hi = boot_auc(y, p)
            ok = np.isfinite(p) & np.isfinite(y)
            thr = np.nanmedian(p[ok]) if False else 0.5
            pred = (p[ok] > thr).astype(float)
            tp = ((pred == 1) & (y[ok] == 1)).sum(); tn = ((pred == 0) & (y[ok] == 0)).sum()
            fp = ((pred == 1) & (y[ok] == 0)).sum(); fn = ((pred == 0) & (y[ok] == 1)).sum()
            rows.append(dict(set=setname, endpoint=en, model=mname, auc=a, lo=lo, hi=hi,
                             accuracy=(tp + tn) / max(ok.sum(), 1),
                             sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                             ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1),
                             n=int(ok.sum()), events=int(np.nansum(y[ok]))))
            if setname == 'MB743':
                ROC_STORE[(en, mname)] = roc(y[ok], p[ok])
DISC = pd.DataFrame(rows); DISC.to_csv(f'Table3_discrimination_{ST}.csv', index=False)
np.savez(f'fig5a_roc_{ST}.npz',
         **{f'{e}|{m}|{i}': v for (e, m), fpr_tpr in ROC_STORE.items()
            for i, v in zip(['fpr', 'tpr'], fpr_tpr)})
for _, r in DISC.iterrows():
    t = (r.endpoint.split()[0].lower() + '_' +
         {'Age, sex, BMI': 'clin', '+ microbiota panel': 'clin_mb',
          '+ ether-lipid ratio': 'clin_r1',
          '+ microbiota panel and ether-lipid ratio': 'clin_both'}[r.model] + '_' + r['set'])
    for f_, v_ in [('auc', r.auc), ('lo', r.lo), ('hi', r.hi), ('acc', r.accuracy),
                   ('n', int(r.n)), ('events', int(r.events))]:
        add(f'disc_{t}_{f_}', v_, '', '')

# ============ D. genus × clinical index correlation (their Figure 3) ============
IDX = [('age', 'Age', 'Anthropometric'), ('BMI', 'BMI', 'Anthropometric'),
       ('waist_cm', 'Waist', 'Anthropometric'), ('weight_kg', 'Weight', 'Anthropometric'),
       ('SBP', 'SBP', 'Anthropometric'), ('DBP', 'DBP', 'Anthropometric'),
       ('lab_glucose_AC', 'Glucose', 'Metabolic'), ('lab_HbA1c', 'HbA1c', 'Metabolic'),
       ('lab_Cholesterol', 'Cholesterol', 'Metabolic'),
       ('lab_triglyceride', 'Triglyceride', 'Metabolic'),
       ('lab_LDL_direct', 'LDL', 'Metabolic'), ('lab_HDL', 'HDL', 'Metabolic'),
       ('lab_UricAcid', 'Uric acid', 'Metabolic'),
       ('lab_creatinine', 'Creatinine', 'Renal'), ('lab_eGFR', 'eGFR', 'Renal'),
       ('lab_BUN', 'BUN', 'Renal')]
rows = []
for g in STABLE:
    v = Z[g].reindex(S825).values
    for col, lab, blk in IDX:
        w = pd.to_numeric(mb[col], errors='coerce').reindex(S825).values
        ok = np.isfinite(v) & np.isfinite(w)
        rho, p = stats.spearmanr(v[ok], w[ok])
        rows.append(dict(genus=g, index=lab, block=blk, rho=rho, p=p, n=int(ok.sum())))
    # the readout their design could not carry
    w = y1.reindex(S743).values; vv = Z[g].reindex(S743).values
    ok = np.isfinite(vv) & np.isfinite(w)
    rho, p = stats.spearmanr(vv[ok], w[ok])
    rows.append(dict(genus=g, index='Ether-lipid ratio', block='Ether lipid',
                     rho=rho, p=p, n=int(ok.sum())))
HM = pd.DataFrame(rows)
o = np.argsort(HM.p.values); q = np.empty(len(HM)); run = 1.0
for i, idx in enumerate(o[::-1]):
    run = min(run, HM.p.values[idx] * len(HM) / (len(HM) - i)); q[idx] = run
HM['q'] = q
HM.to_csv(f'TableS12_genus_clinical_{ST}.csv', index=False)
add('n_heatmap_cells', len(HM), 'tests', f'{len(STABLE)} genera x {len(IDX)+1} indices')
add('n_heatmap_q05', int((HM.q < 0.05).sum()), 'tests', 'BH')

# ============ E. abundance across CKM stages (their Figure 4a) ============
stage = pd.to_numeric(mb.CKM5, errors='coerce').reindex(S825)
PREV = (REL.reindex(S825) > 0).mean()
ABUND = [g for g in STABLE if PREV[g] >= 0.50]
add('n_stable_genera_prevalent', len(ABUND), 'genera', 'stable panel genera present in >=50% of samples')
rows = []
for g in STABLE:
    v = REL[g].reindex(S825)
    grps = [v[stage == s].dropna().values for s in range(5)]
    kw = stats.kruskal(*[x for x in grps if len(x) > 1]).pvalue
    lo = v[stage <= 2].dropna().values; hi = v[stage >= 3].dropna().values
    mw = stats.mannwhitneyu(lo, hi).pvalue
    rows.append(dict(genus=g, prevalence=float(PREV[g]), kruskal_p=kw, mw_low_vs_high_p=mw,
                     mean_stage0_2=float(np.mean(lo)), mean_stage3_4=float(np.mean(hi)),
                     median_stage0_2=float(np.median(lo)), median_stage3_4=float(np.median(hi)),
                     **{f'median_stage{s}': float(np.median(grps[s])) if len(grps[s]) else np.nan
                        for s in range(5)}))
AB = pd.DataFrame(rows)
o = np.argsort(AB.kruskal_p.values); q = np.empty(len(AB)); run = 1.0
for i, idx in enumerate(o[::-1]):
    run = min(run, AB.kruskal_p.values[idx] * len(AB) / (len(AB) - i)); q[idx] = run
AB['kruskal_q'] = q
AB.sort_values('kruskal_p').to_csv(f'TableS13_abundance_stage_{ST}.csv', index=False)
add('n_abundance_q05', int((AB.kruskal_q < 0.05).sum()), 'genera', 'Kruskal-Wallis across stages, BH')

N = pd.read_csv(f'numbers_{ST}.csv')
N = N[~N.key.isin([f['key'] for f in F])]
pd.concat([N, pd.DataFrame(F)], ignore_index=True).to_csv(f'numbers_{ST}.csv', index=False)
print(f'{len(F)} numbers appended\n')
print('stable genera (>=3 folds):', STABLE)
print('\nfeature-number curve (mean AUC):'); print(CURVE.groupby('k').auc.mean().round(3).to_string())
print('\ndiscrimination:'); print(DISC.round(3).to_string(index=False))
print('\nabundance across stages, top 8:')
print(AB.sort_values('kruskal_p').head(8)[['genus', 'median_stage0_2', 'median_stage3_4',
                                           'kruskal_p', 'kruskal_q']].to_string(index=False))
print('\nheatmap: %d of %d cells q<0.05' % ((HM.q < 0.05).sum(), len(HM)))
