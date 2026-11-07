"""Rebuild the eight ratio phenotypes WITHOUT the genotype gate — 20260911_v54.

pheno_final_20260822_v25 was built on metabolome ∩ genotype (n = 1,998) because every
model carried ten genetic principal components. The microbiome sample is being relaxed to
every participant with 16S, so the phenotypes have to exist for the whole metabolome set.

Construction follows the Methods exactly: sum the constituent species, then divide; take
the natural log; centre within metabolomics run year at the level of the individual
measurement; average across repeat measurements within a participant. The rank-based
inverse normal transform is applied downstream, on whatever analysis sample is in use.

The reconstruction is validated against the released file on the 1,998 overlap before
anything else is allowed to use it.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re, sys

UP = '/mnt/user-data/uploads/CKM+microbiome'
ST = '20260911_v54'

D = pd.read_csv(f'{UP}/analysis/04_data_matrices/p180_all.csv', low_memory=False)
D['sid'] = D.subject_base.astype(str)
cols = list(D.columns)

PCae  = [c for c in cols if re.match(r'^PC ae ', c)]
PCaa  = [c for c in cols if re.match(r'^PC aa ', c)]
LYSO  = [c for c in cols if re.match(r'^lysoPC ', c)]
SM    = [c for c in cols if re.match(r'^SM ', c)]

def one(name):
    if name not in cols: sys.exit(f'missing metabolite column: {name}')
    return [name]

# numerator / denominator species lists for each pre-specified ratio
RATIO = {
    'R1_PCae_lysoPC_z':        (PCae,                    LYSO),
    'R2_PCae362_361_z':        (one('PC ae C36:2'),      one('PC ae C36:1')),
    'R3_SM_over_PC_z':         (SM,                      PCaa + PCae),
    'R4_SM240_160_z':          (one('SM C24:0'),         one('SM C16:0')),
    'R5_lysoPC204_PCaa384_z':  (one('lysoPC a C20:4'),   one('PC aa C38:4')),
    'R6_PCaa384_383_z':        (one('PC aa C38:4'),      one('PC aa C38:3')),
    'R7_C16_C0_z':             (one('C16'),              one('C0')),
    'R8_C2_C0_z':              (one('C2'),               one('C0')),
}

num = lambda cs: D[cs].apply(pd.to_numeric, errors='coerce')
year = pd.to_numeric(D.test_year, errors='coerce')

meas = pd.DataFrame({'sid': D.sid, 'year': year})
for k, (n_, d_) in RATIO.items():
    r = num(n_).sum(axis=1, min_count=1) / num(d_).sum(axis=1, min_count=1)
    lr = np.log(r.where(r > 0))
    meas[k] = lr - lr.groupby(year).transform('mean')      # centre within run year

PHE = meas.groupby('sid')[list(RATIO)].mean().reset_index().rename(columns={'sid': 'subject_id'})
PHE['n_draws'] = meas.groupby('sid').size().reindex(PHE.subject_id).values

# age and sex, averaged / taken from the metabolomics file itself
AGE = pd.to_numeric(D['年紀'], errors='coerce').groupby(D.sid).mean()
SEXRAW = D['性別'].astype(str).str.strip()
PHE['age_p180'] = PHE.subject_id.map(AGE)
PHE['sex_p180'] = PHE.subject_id.map(
    SEXRAW.groupby(D.sid).first().map(lambda v: 1.0 if v in ('1', 'M', '男') else
                                      (0.0 if v in ('0', '2', 'F', '女') else np.nan)))

# ---------------- validation against the released file ----------------
REL = pd.read_csv(f'{UP}/analysis/pheno_final_20260822_v25.csv')
REL['subject_id'] = REL.subject_id.astype(str)
M = REL[['subject_id'] + list(RATIO)].merge(PHE, on='subject_id', suffixes=('_rel', '_new'))
print(f'validation overlap: {len(M)} participants (released file has {len(REL)})')
bad = []
for k in RATIO:
    a, b = M[f'{k}_rel'], M[f'{k}_new']
    ok = a.notna() & b.notna()
    r = np.corrcoef(a[ok], b[ok])[0, 1]
    print(f'  {k:26s} r = {r:.4f}   n = {ok.sum()}')
    if r < 0.99: bad.append(k)
if bad:
    sys.exit(f'reconstruction does not reproduce the released phenotypes: {bad}')

PHE.to_csv(f'pheno_all_{ST}.csv', index=False)
print(f'\nwritten pheno_all_{ST}.csv: {len(PHE)} participants, '
      f'{int(PHE.n_draws.sum())} measurements')
