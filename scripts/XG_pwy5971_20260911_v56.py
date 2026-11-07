"""Is the palmitate-biosynthesis signal a finding in its own right? — 20260911_v56.

PWY-5971 is the single MetaCyc pathway that survives adjustment for reference-genome
coverage against both the microbiota exposure and the host ether-lipid readout. Before it
can be reported as a finding rather than an observation, it has to answer the three
questions a reviewer will ask.

  1  Does it act at the ether branch specifically, or on phosphatidylcholine generally?
     The same class dissociation the paper is built on is applied to it: PC ae, PC aa,
     their ratio, and lysoPC.
  2  Does it add anything to the microbiota panel, or is it the panel in another guise?
     The ether-lipid model is refitted with both in it.
  3  Does it sit on the path between the microbiota and the host readout? Bootstrap
     mediation of microbiota panel -> PWY-5971 -> ether-lipid ratio.

Every model adjusts for age, sex, body-mass index, sequencing batch and the weighted
nearest-sequenced taxon index, so nothing here can be a coverage artefact.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from scipy import stats
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

A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
mb = A.set_index('subject_id'); s2subj = dict(zip(A.sample_id, A.subject_id))
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
S = [s for s in A.subject_id if s in set(PHE.subject_id)]
add('n_pwy5971_set', len(S), 'people', '')

PA = pd.read_csv(PIC + 'Galaxy16-[PICRUSt2 Full pipeline on dataset 11 and 12_ '
                       'Pathway abundances].tabular', sep='\t', index_col=0)
PA = PA[[c for c in PA.columns if c in s2subj]]
PA.columns = [s2subj[c] for c in PA.columns]
PC_ = np.log(PA.T + 0.5); PC_ = PC_.sub(PC_.mean(1), axis=0)
pz = ((PC_ - PC_.mean()) / PC_.std())['PWY-5971'].reindex(S).values
NS = pd.read_csv(PIC + 'Galaxy21-[EC_weighted_nsti].tabular', sep='\t')
NS['subject_id'] = NS['sample'].map(s2subj)
nsti = rint(NS.dropna(subset=['subject_id']).set_index('subject_id')
            .weighted_NSTI.reindex(S).values)
SC = pd.read_csv('panel_scores_20260911_v54.csv'); SC['subject_id'] = SC.subject_id.astype(str)
panel = SC.set_index('subject_id').panel_oof.reindex(S).values

age = pd.to_numeric(mb.age, errors='coerce').reindex(S).values
sex = pd.to_numeric(mb.sex, errors='coerce').reindex(S).values
bmi = pd.to_numeric(mb.BMI, errors='coerce').reindex(S).values
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float).reindex(S).values

# ---------- host lipid phenotypes, built exactly as in the main analysis ----------
RAW = pd.read_csv(f'{UP}/analysis/04_data_matrices/p180_all.csv', low_memory=False)
RAW['sid'] = RAW.subject_base.astype(str); yr = pd.to_numeric(RAW.test_year, errors='coerce')
PCAE = [c for c in RAW.columns if re.match(r'^PC ae ', c)]
PCAA = [c for c in RAW.columns if re.match(r'^PC aa ', c)]
LYSO = [c for c in RAW.columns if re.match(r'^lysoPC ', c)]
num = lambda cs: RAW[cs].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
t_ae, t_aa, t_ly = num(PCAE), num(PCAA), num(LYSO)
def pheno(expr):
    lv = np.log(expr.where(expr > 0)); lv = lv - lv.groupby(yr).transform('mean')
    v = lv.groupby(RAW.sid).mean()
    return pd.Series(rint(v.reindex(PHE.subject_id).values),
                     index=PHE.subject_id).reindex(S).values
LIP = {'Ether-lipid ratio (PC ae / lysoPC)': pheno(t_ae / t_ly),
       'PC ae / PC aa ratio': pheno(t_ae / t_aa),
       'Total PC ae': pheno(t_ae), 'Total PC aa': pheno(t_aa), 'Total lysoPC': pheno(t_ly)}

def fit(y, X0):
    X = np.column_stack([X0, np.ones(len(S)), age, sex, bmi, BAT, nsti])
    y = np.asarray(y, float); ok = np.isfinite(X).all(1) & np.isfinite(y)
    Xo = X[ok]; nc = np.shape(X0)[1] if np.ndim(X0) > 1 else 1
    keep = np.r_[np.ones(nc, bool), Xo[:, nc:].std(0) > 1e-10]; keep[nc] = True
    Xo = Xo[:, keep]
    b, se = hc3(Xo, y[ok])
    df = ok.sum() - Xo.shape[1]
    return [(b[i], se[i], 2 * stats.t.sf(abs(b[i] / se[i]), df)) for i in range(nc)] + \
           [int(ok.sum())]

# ---------- 1. class specificity ----------
rows = []
for nm, y in LIP.items():
    (b, se, p), n = fit(y, pz)[0], fit(y, pz)[-1]
    rows.append(dict(phenotype=nm, beta=b, se=se, p=p, n=n))
    k = nm.split('(')[0].strip().lower().replace(' / ', '_over_').replace(' ', '_')
    add(f'pwy5971_{k}_beta', b, '', ''); add(f'pwy5971_{k}_p', p, '', '')
CLS = pd.DataFrame(rows)
print('1. Does PWY-5971 act at the ether branch? (adjusted for NSTI)')
print(CLS.to_string(index=False))

# ---------- 2. independent of the microbiota panel? ----------
y = LIP['Ether-lipid ratio (PC ae / lysoPC)']
joint = fit(y, np.column_stack([pz, panel]))
(bp, sp, pp), (bm, sm_, pm), n = joint[0], joint[1], joint[2]
alone_p = fit(y, pz)[0]; alone_m = fit(y, panel)[0]
JOINT = pd.DataFrame([
    dict(model='PWY-5971 alone', term='PWY-5971', beta=alone_p[0], se=alone_p[1], p=alone_p[2]),
    dict(model='Microbiota panel alone', term='panel', beta=alone_m[0], se=alone_m[1], p=alone_m[2]),
    dict(model='Both in the model', term='PWY-5971', beta=bp, se=sp, p=pp),
    dict(model='Both in the model', term='panel', beta=bm, se=sm_, p=pm)])
for k, v in [('joint_pwy_beta', bp), ('joint_pwy_p', pp), ('joint_panel_beta', bm),
             ('joint_panel_p', pm), ('joint_n', n),
             ('pwy_attenuation_pct', round(100 * (1 - bp / alone_p[0]))),
             ('panel_attenuation_pct', round(100 * (1 - bm / alone_m[0])))]:
    add(k, v, '', '')
print('\n2. Is it just the panel in another guise?')
print(JOINT.to_string(index=False))
print(f'   PWY-5971 attenuates {100*(1-bp/alone_p[0]):.0f}% when the panel is added; '
      f'the panel attenuates {100*(1-bm/alone_m[0]):.0f}% when PWY-5971 is added (n = {n})')

# ---------- 3. mediation: panel -> PWY-5971 -> ether-lipid ratio ----------
rng = np.random.default_rng(20260911)
Cv = np.column_stack([np.ones(len(S)), age, sex, bmi, BAT, nsti])
ok = np.isfinite(Cv).all(1) & np.isfinite(pz) & np.isfinite(panel) & np.isfinite(y)
Cvo, pzo, pao, yo = Cv[ok], pz[ok], panel[ok], y[ok]
Xa = np.column_stack([pao, Cvo]); Xb = np.column_stack([pzo, pao, Cvo]); Xt = Xa
a_ = lambda Xm, ym: np.linalg.lstsq(Xm, ym, rcond=None)[0][0]
ab = a_(Xa, pzo) * a_(Xb, yo); tot = a_(Xt, yo)
bs = []
for _ in range(2000):
    j = rng.integers(0, ok.sum(), ok.sum())
    try: bs.append(a_(Xa[j], pzo[j]) * a_(Xb[j], yo[j]))
    except Exception: pass
bs = np.array([v for v in bs if np.isfinite(v)]); lo, hi = np.percentile(bs, [2.5, 97.5])
for k, v in [('med5971_indirect', ab), ('med5971_lo', lo), ('med5971_hi', hi),
             ('med5971_total', tot), ('med5971_prop', ab / tot), ('med5971_n', int(ok.sum()))]:
    add(k, v, '', 'bootstrap 2000')
print(f'\n3. Mediation  panel -> PWY-5971 -> ether-lipid ratio')
print(f'   indirect {ab:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}); total {tot:+.4f}; '
      f'{100*ab/tot:.0f}% of the total effect; n = {ok.sum()}')

CLS.to_csv(f'TableS24_pwy5971_class_{ST}.csv', index=False)
JOINT.to_csv(f'TableS25_pwy5971_joint_{ST}.csv', index=False)
pd.DataFrame(F).to_csv(f'numbers_pwy5971_{ST}.csv', index=False)
