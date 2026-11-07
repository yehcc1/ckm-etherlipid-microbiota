"""Predicted microbial pathways against circulating metabolites — 20260911_v56.

The question this answers is the one the TMAO literature poses: does a microbial reaction
change a metabolite in the host's blood, and does that metabolite track disease? Three
analyses, in decreasing order of how much they can be believed.

  A  Matched pairs (pre-specifiable).  For every non-lipid metabolite the p180 panel
     measures, take every MetaCyc pathway whose name contains that metabolite. The
     substrate or product of the reaction is then measured directly in plasma, so the
     direction is predicted in advance. 60 pairs over 23 metabolites.
  B  Broad scan.  The 69 pathways that survived NSTI adjustment for the microbiota panel,
     against all 30 non-lipid metabolites.
  C  The chain.  Any pathway -> metabolite association is carried forward to
     metabolite -> clinical outcome in the full metabolome sample, and the pathway ->
     metabolite -> outcome indirect effect is bootstrapped.

Every pathway model is run twice, with and without the weighted nearest-sequenced taxon
index, because that is what separates biology from reference-genome coverage here.
"""
import warnings; warnings.filterwarnings('ignore')
import gzip
import numpy as np, pandas as pd, re
from scipy import stats
import statsmodels.api as sm
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
PIC = f'{UP}/PICRUSt2 Galaxy/'
ST = '20260911_v56'
F = []
add = lambda k, v, u='', s='': F.append(dict(key=k, value=v, unit=u, source=s))
rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))
def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty(len(p)); run = 1.0
    for i, idx in enumerate(o[::-1]):
        run = min(run, p[idx] * len(p) / (len(p) - i)); q[idx] = run
    return q

# ---------------- metabolite phenotypes ----------------
RAW = pd.read_csv(f'{UP}/analysis/04_data_matrices/p180_all.csv', low_memory=False)
RAW['sid'] = RAW.subject_base.astype(str)
yr = pd.to_numeric(RAW.test_year, errors='coerce')
MET = ['Ala', 'Arg', 'Asn', 'Asp', 'Cit', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys',
       'Met', 'Orn', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val', 'ADMA', 'Kynurenine',
       'Putrescine', 'Sarcosine', 'SDMA', 'Serotonin', 't4-OH-Pro', 'Taurine']
mrows = {}
for m in MET:
    v = pd.to_numeric(RAW[m], errors='coerce')
    lv = np.log(v.where(v > 0)); lv = lv - lv.groupby(yr).transform('mean')
    mrows[m] = lv.groupby(RAW.sid).mean()
MET_DF = pd.DataFrame(mrows)
MET_Z = MET_DF.apply(lambda c: pd.Series(rint(c.values), index=c.index))
add('n_metabolites_tested', len(MET), 'metabolites', 'p180 amino acids and biogenic amines')

# ---------------- samples, pathways, covariates ----------------
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
from stage_v10 import master_stage
A = A.assign(CKM5=master_stage(A, 'XF MB825'))
mb = A.set_index('subject_id'); S825 = list(A.subject_id)
S = [s for s in S825 if s in MET_Z.index]                 # pathway + metabolite = 743
add('n_pathway_metabolite_set', len(S), 'people', '16S with metabolomics')
s2subj = dict(zip(A.sample_id, A.subject_id))

PA = pd.read_csv(PIC + 'Galaxy16-[PICRUSt2 Full pipeline on dataset 11 and 12_ '
                       'Pathway abundances].tabular', sep='\t', index_col=0)
PA = PA[[c for c in PA.columns if c in s2subj]]
PA.columns = [s2subj[c] for c in PA.columns]
PA = PA.T.reindex(S)
PC = np.log(PA + 0.5); PC = PC.sub(PC.mean(1), axis=0)
PZ = (PC - PC.mean()) / PC.std()
NAME = dict(l.split('\t', 1) for l in gzip.open(
    'metacyc_names_from_S17.txt.gz', 'rt', encoding='utf-8', errors='replace'
).read().rstrip('\n').split('\n'))

NS = pd.read_csv(PIC + 'Galaxy21-[EC_weighted_nsti].tabular', sep='\t')
NS['subject_id'] = NS['sample'].map(s2subj)
nsti = rint(NS.dropna(subset=['subject_id']).set_index('subject_id')
            .weighted_NSTI.reindex(S).values)

age = pd.to_numeric(mb.age, errors='coerce').reindex(S).values
sex = pd.to_numeric(mb.sex, errors='coerce').reindex(S).values
bmi = pd.to_numeric(mb.BMI, errors='coerce').reindex(S).values
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float).reindex(S).values

def fit(y, x, extra=None):
    cov = [np.ones(len(S)), age, sex, bmi, BAT]
    if extra is not None: cov.append(np.asarray(extra, float))
    X = np.column_stack([np.asarray(x, float)] + cov)
    y = np.asarray(y, float); ok = np.isfinite(X).all(1) & np.isfinite(y)
    Xo = X[ok]; keep = np.r_[True, Xo[:, 1:].std(0) > 1e-10]; keep[1] = True
    Xo = Xo[:, keep]
    b, se = hc3(Xo, y[ok])
    return b[0], se[0], 2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - Xo.shape[1]), int(ok.sum())

MZ = MET_Z.reindex(S)

# ================= A. matched pathway-metabolite pairs =================
KEY = {'Ala': 'L-alanine', 'Arg': 'L-arginine', 'Asn': 'L-asparagine', 'Asp': 'L-aspartate',
       'Cit': 'L-citrulline', 'Gln': 'L-glutamine', 'Glu': 'L-glutamate', 'Gly': 'glycine',
       'His': 'L-histidine', 'Ile': 'L-isoleucine', 'Leu': 'L-leucine', 'Lys': 'L-lysine',
       'Met': 'L-methionine', 'Orn': 'L-ornithine', 'Phe': 'L-phenylalanine',
       'Pro': 'L-proline', 'Ser': 'L-serine', 'Thr': 'L-threonine', 'Trp': 'L-tryptophan',
       'Tyr': 'L-tyrosine', 'Val': 'L-valine', 'Kynurenine': 'kynurenin',
       'Putrescine': 'putrescine', 'Sarcosine': 'sarcosine', 'Serotonin': 'serotonin',
       'Taurine': 'taurine', 't4-OH-Pro': 'hydroxyproline'}
pairs = [(m, p_) for m, k in KEY.items() for p_ in PZ.columns if k.lower() in NAME[p_].lower()]
rows = []
for m, p_ in pairs:
    b, se, p, n = fit(MZ[m].values, PZ[p_].values)
    b2, se2, p2, _ = fit(MZ[m].values, PZ[p_].values, extra=nsti)
    direction = 'degradation' if re.search(r'degrad|catabol|ferment', NAME[p_], re.I) else \
                ('biosynthesis' if re.search(r'biosynth|synth', NAME[p_], re.I) else 'other')
    rows.append(dict(metabolite=m, pathway=p_, pathway_name=NAME[p_], kind=direction,
                     beta=b, se=se, p=p, n=n, beta_nsti_adj=b2, p_nsti_adj=p2))
PAIR = pd.DataFrame(rows)
PAIR['q'] = bh(PAIR.p.values); PAIR['q_nsti_adj'] = bh(PAIR.p_nsti_adj.values)
PAIR.sort_values('p').to_csv(f'TableS20_pathway_metabolite_pairs_{ST}.csv', index=False)
add('n_matched_pairs', len(PAIR), 'pairs', '')
add('n_matched_pairs_q05', int((PAIR.q < 0.05).sum()), 'pairs', '')
add('n_matched_pairs_q05_nsti', int(((PAIR.q < 0.05) & (PAIR.q_nsti_adj < 0.05)).sum()),
    'pairs', '')
print(f'A. matched pairs: {len(PAIR)} tested, {(PAIR.q<0.05).sum()} at q<0.05, '
      f'{((PAIR.q<0.05)&(PAIR.q_nsti_adj<0.05)).sum()} survive NSTI')
print(PAIR.sort_values('p').head(14)[['metabolite', 'pathway', 'pathway_name', 'kind',
                                      'beta', 'q', 'q_nsti_adj']].to_string(index=False))

# ================= B. surviving panel pathways x all metabolites =================
M17 = pd.read_csv('TableS17_metacyc_enrichment_20260911_v55.csv')
SURV = M17[(M17.exposure == 'Microbiota panel') & (M17.q < 0.05) &
           (M17.q_nsti_adj < 0.05)].pathway.tolist()
rows = []
for p_ in SURV:
    for m in MET:
        b, se, p, n = fit(MZ[m].values, PZ[p_].values)
        b2, se2, p2, _ = fit(MZ[m].values, PZ[p_].values, extra=nsti)
        rows.append(dict(pathway=p_, pathway_name=NAME[p_], metabolite=m, beta=b, se=se,
                         p=p, n=n, beta_nsti_adj=b2, p_nsti_adj=p2))
BROAD = pd.DataFrame(rows)
BROAD['q'] = bh(BROAD.p.values); BROAD['q_nsti_adj'] = bh(BROAD.p_nsti_adj.values)
BROAD.sort_values('p').to_csv(f'TableS21_pathway_metabolite_broad_{ST}.csv', index=False)
add('n_broad_tests', len(BROAD), 'tests', f'{len(SURV)} surviving pathways x {len(MET)} metabolites')
add('n_broad_q05', int((BROAD.q < 0.05).sum()), 'tests', '')
add('n_broad_q05_nsti', int(((BROAD.q < 0.05) & (BROAD.q_nsti_adj < 0.05)).sum()), 'tests', '')
print(f'\nB. broad scan: {len(BROAD)} tests, {(BROAD.q<0.05).sum()} at q<0.05, '
      f'{((BROAD.q<0.05)&(BROAD.q_nsti_adj<0.05)).sum()} survive NSTI')
print(BROAD.sort_values('p').head(16)[['pathway', 'pathway_name', 'metabolite', 'beta',
                                       'q', 'q_nsti_adj']].to_string(index=False))

# acetylene degradation specifically, because it is the largest pathway effect
ACE = BROAD[BROAD.pathway == 'P161-PWY'].sort_values('p')
ACE.to_csv(f'acetylene_vs_metabolites_{ST}.csv', index=False)
add('acetylene_n_q05', int((ACE.q < 0.05).sum()), 'metabolites', '')
print('\nAcetylene degradation (P161-PWY) against every metabolite, top 8:')
print(ACE.head(8)[['metabolite', 'beta', 'p', 'q', 'q_nsti_adj']].to_string(index=False))

# ================= C. the chain: metabolite -> clinical outcome =================
C = pd.read_csv(f'{UP}/CKM_clinical_for_microbiome_20260819_V6.csv', low_memory=False)
C['subject_id'] = C.subject_id.astype(str)
C = C.sort_values('visit_no').drop_duplicates('subject_id').set_index('subject_id')
C = C.assign(CKM5=master_stage(C, 'XF clinical'))
SM = [s for s in MET_Z.index if s in C.index]
add('n_metabolite_clinical', len(SM), 'people', '')
acl = pd.to_numeric(C.age, errors='coerce'); scl = pd.to_numeric(C.sex, errors='coerce')
bcl = pd.to_numeric(C.BMI, errors='coerce')
def fit_cl(y, x, binary=False):
    Cv = np.column_stack([np.ones(len(SM)), acl.reindex(SM).values, scl.reindex(SM).values,
                          bcl.reindex(SM).values])
    X = np.column_stack([np.asarray(x, float), Cv]); y = np.asarray(y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if binary:
        r = sm.GLM(y[ok], X[ok], family=sm.families.Binomial()).fit(cov_type='HC3')
        return r.params[0], r.bse[0], r.pvalues[0], int(ok.sum())
    b, se = hc3(X[ok], y[ok])
    return b[0], se[0], 2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - X.shape[1]), int(ok.sum())
OUT = [('log(triglycerides)', rint(np.log(pd.to_numeric(C.lab_triglyceride, errors='coerce')
                                          .reindex(SM))), False),
       ('HbA1c', rint(pd.to_numeric(C.lab_HbA1c, errors='coerce').reindex(SM).values), False),
       ('Diabetes', pd.to_numeric(C.C_DM, errors='coerce').reindex(SM).values, True),
       ('CKM stage (5-level)', rint(pd.to_numeric(C.CKM5, errors='coerce').reindex(SM).values), False),
       ('eGFR', rint(pd.to_numeric(C.lab_eGFR, errors='coerce').reindex(SM).values), False)]
MZ_all = MET_Z.reindex(SM)
rows = []
for m in MET:
    for on, y, bi in OUT:
        b, se, p, n = fit_cl(y, MZ_all[m].values, binary=bi)
        rows.append(dict(metabolite=m, outcome=on, beta=b, se=se, p=p, n=n,
                         scale='log-odds per SD' if bi else 'SD per SD'))
CLIN = pd.DataFrame(rows); CLIN['q'] = bh(CLIN.p.values)
CLIN.sort_values('p').to_csv(f'TableS22_metabolite_clinical_{ST}.csv', index=False)
add('n_metabolite_clinical_tests', len(CLIN), 'tests', '')
add('n_metabolite_clinical_q05', int((CLIN.q < 0.05).sum()), 'tests', '')
print(f'\nC. metabolite -> clinical: {len(CLIN)} tests, {(CLIN.q<0.05).sum()} at q<0.05 '
      f'(n up to {CLIN.n.max():,})')

# complete chains: pathway -> metabolite (survives NSTI) -> outcome (q<0.05)
hits = pd.concat([PAIR[(PAIR.q < 0.05) & (PAIR.q_nsti_adj < 0.05)]
                  [['pathway', 'pathway_name', 'metabolite', 'beta', 'q', 'q_nsti_adj']],
                  BROAD[(BROAD.q < 0.05) & (BROAD.q_nsti_adj < 0.05)]
                  [['pathway', 'pathway_name', 'metabolite', 'beta', 'q', 'q_nsti_adj']]
                  ]).drop_duplicates(['pathway', 'metabolite'])
chain = hits.merge(CLIN[CLIN.q < 0.05][['metabolite', 'outcome', 'beta', 'q', 'scale']],
                   on='metabolite', suffixes=('_path_met', '_met_out'))
chain.to_csv(f'TableS23_complete_chains_{ST}.csv', index=False)
add('n_complete_chains', len(chain), 'chains', 'pathway -> metabolite -> outcome, both q<0.05')
add('n_chain_metabolites', int(chain.metabolite.nunique()), 'metabolites', '')
print(f'\nComplete chains (pathway -> metabolite -> outcome, every link q < 0.05): {len(chain)}')
if len(chain):
    print(chain.sort_values('q_met_out').head(20).to_string(index=False))

pd.DataFrame(F).to_csv(f'numbers_pathmet_{ST}.csv', index=False)
