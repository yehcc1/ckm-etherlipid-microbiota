"""Does richness show the same class dissociation as the panel? — 20260912_v60.

The whole argument of the paper rests on ether phosphatidylcholines moving while diacyl
phosphatidylcholines do not. If alpha diversity produced the same dissociation, the panel
would be a richness proxy dressed up; if richness moves both classes, or neither ratio, then
the dissociation is specific to composition. The class phenotypes are rebuilt exactly as in
X2_core_825 (sum, log, residualise on test year, average within participant, rank-INT) and
the same HC3 models are used.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from scipy import stats
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'
ST = '20260913_v60'
rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))

def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))

A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = (A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
        (~A0.subject_id.str.startswith('J'))]
     .drop_duplicates('subject_id').set_index('subject_id'))
AL = pd.read_csv(f'{HUB}/alpha_diversity.csv').set_index('sample_id').reindex(A.sample_id.values)
AL.index = A.index
PS = pd.read_csv('panel_scores_20260911_v54.csv').set_index('subject_id')
PS.index = PS.index.astype(str)
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
S743 = [s for s in A.index if s in set(PHE.subject_id)]

RAW = pd.read_csv(f'{UP}/analysis/04_data_matrices/p180_all.csv', low_memory=False)
RAW['sid'] = RAW.subject_base.astype(str); yr = pd.to_numeric(RAW.test_year, errors='coerce')
PCAE = [c for c in RAW.columns if re.match(r'^PC ae ', c)]
LYSO = [c for c in RAW.columns if re.match(r'^lysoPC ', c)]
PCAA = [c for c in RAW.columns if re.match(r'^PC aa ', c)]
num = lambda cols: RAW[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
t_ae, t_aa, t_ly = num(PCAE), num(PCAA), num(LYSO)

def class_pheno(expr):
    lr = np.log(expr.where(expr > 0)); lr = lr - lr.groupby(yr).transform('mean')
    v = lr.groupby(RAW.sid).mean()
    return pd.Series(rint(v.reindex(PHE.subject_id).values),
                     index=PHE.subject_id).reindex(S743).values

CLS = {'PC ae / PC aa': t_ae / t_aa, 'Total PC ae': t_ae, 'Total PC aa': t_aa,
       'Total lysoPC': t_ly, 'PC ae / lysoPC (R1)': t_ae / t_ly}

BAT = pd.get_dummies(A.batch, drop_first=True).astype(float)
COV = pd.DataFrame({'age': pd.to_numeric(A.age, errors='coerce'),
                    'sex': pd.to_numeric(A.sex, errors='coerce'),
                    'bmi': pd.to_numeric(A.BMI, errors='coerce'),
                    'logdepth': np.log(AL.total_reads.astype(float))}, index=A.index).join(BAT)
EXP = {'Observed ASVs': AL.n_asv.astype(float), 'Shannon': AL.shannon.astype(float),
       'CV panel': PS.panel_oof.reindex(A.index),
       'Pre-specified score': PS.score_prior.reindex(A.index)}
cols = ['age', 'sex', 'bmi', 'logdepth'] + list(BAT.columns)

rows = []
for nm, expr in CLS.items():
    y = pd.Series(class_pheno(expr), index=S743)
    for xn in EXP:
        d = pd.DataFrame({'y': y, 'x': stats.zscore(EXP[xn].reindex(S743).values,
                                                    nan_policy='omit')},
                         index=S743).join(COV).dropna()
        X = np.column_stack([np.ones(len(d)), d.x.values, d[cols].values])
        b, se = hc3(X, d.y.values)
        rows.append(dict(phenotype=nm, exposure=xn, beta=b[1], se=se[1],
                         p=2 * stats.norm.sf(abs(b[1] / se[1])), n=len(d)))
OUT = pd.DataFrame(rows)
OUT.to_csv(f'TableS28b_alpha_class_specificity_{ST}.csv', index=False)
pd.set_option('display.width', 200)
print(OUT.pivot(index='phenotype', columns='exposure', values=['beta', 'p'])
      .to_string(float_format='%.4g'))
