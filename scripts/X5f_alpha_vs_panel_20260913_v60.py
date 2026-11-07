"""Is the microbiota panel just richness? — 20260912_v60.

The diversity run raised the question directly: observed ASV count correlates with the
cross-validated panel at rho = 0.56. If the panel's association with the host ether-lipid
ratio is really an alpha-diversity association, that has to be found now rather than by a
reviewer. Three tests, all in the 743 participants with metabolomics:

  1. each alpha metric against the ether-lipid ratio on its own, with and without read depth;
  2. the panel against the ether-lipid ratio with the alpha metric in the same model;
  3. the same for the pre-specified three-genus score, which is fixed and cannot have been
     selected for richness.

Covariates throughout: age, sex, body-mass index, 16S batch (and log read depth where stated).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'
ST = '20260913_v60'

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
rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))
PHE['R1_int'] = rint(PHE['R1_PCae_lysoPC_z'].values)
R1 = PHE.set_index('subject_id').R1_int

def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))

BAT = pd.get_dummies(A.batch, drop_first=True).astype(float)
BASE = pd.DataFrame({'age': pd.to_numeric(A.age, errors='coerce'),
                     'sex': pd.to_numeric(A.sex, errors='coerce'),
                     'bmi': pd.to_numeric(A.BMI, errors='coerce'),
                     'logdepth': np.log(AL.total_reads.astype(float))}, index=A.index).join(BAT)
EXP = {'Observed ASVs': AL.n_asv.astype(float), 'Shannon': AL.shannon.astype(float),
       'Pielou evenness': AL.evenness.astype(float),
       'CV panel': PS.panel_oof.reindex(A.index), 'Pre-specified score': PS.score_prior.reindex(A.index)}

def model(terms, depth):
    cols = ['age', 'sex', 'bmi'] + (['logdepth'] if depth else []) + list(BAT.columns)
    d = pd.DataFrame({'y': R1.reindex(A.index).values}, index=A.index)
    for i, t in enumerate(terms):
        d[f'x{i}'] = stats.zscore(EXP[t].values, nan_policy='omit')
    d = d.join(BASE[cols]).dropna()
    X = np.column_stack([np.ones(len(d))] + [d[f'x{i}'].values for i in range(len(terms))] +
                        [d[cols].values])
    b, se = hc3(X, d.y.values)
    return [dict(term=t, beta=b[i + 1], se=se[i + 1],
                 p=2 * stats.norm.sf(abs(b[i + 1] / se[i + 1])), n=len(d),
                 model=' + '.join(terms) + (' + depth' if depth else ''))
            for i, t in enumerate(terms)]

rows = []
for t in ['Observed ASVs', 'Shannon', 'Pielou evenness', 'CV panel', 'Pre-specified score']:
    for depth in (False, True):
        rows += model([t], depth)
for a in ['Observed ASVs', 'Shannon']:
    for e in ['CV panel', 'Pre-specified score']:
        rows += model([e, a], True)
OUT = pd.DataFrame(rows)[['model', 'term', 'beta', 'se', 'p', 'n']]
OUT.to_csv(f'TableS28_panel_vs_alpha_{ST}.csv', index=False)

# how much of the panel's effect is shared with richness
solo = OUT[(OUT.model == 'CV panel + depth') & (OUT.term == 'CV panel')].iloc[0]
joint = OUT[(OUT.model == 'CV panel + Observed ASVs + depth') & (OUT.term == 'CV panel')].iloc[0]
solo_p = OUT[(OUT.model == 'Pre-specified score + depth') & (OUT.term == 'Pre-specified score')].iloc[0]
joint_p = OUT[(OUT.model == 'Pre-specified score + Observed ASVs + depth') &
              (OUT.term == 'Pre-specified score')].iloc[0]
pd.set_option('display.width', 200)
print(OUT.to_string(index=False, float_format='%.4g'))
print(f'\nCV panel on the ether-lipid ratio: {solo.beta:+.3f} alone, {joint.beta:+.3f} with '
      f'observed ASVs in the model ({100*(1-joint.beta/solo.beta):.0f}% attenuation, '
      f'P = {joint.p:.3g})')
print(f'Pre-specified score: {solo_p.beta:+.3f} alone, {joint_p.beta:+.3f} with observed ASVs '
      f'({100*(1-joint_p.beta/solo_p.beta):.0f}% attenuation, P = {joint_p.p:.3g})')
