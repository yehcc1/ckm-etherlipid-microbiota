"""Genetics as a supplementary sensitivity analysis — 20260911_v54.

Genotype is no longer an entry requirement, so this script answers the two questions that
removing it raises, and nothing more. Both belong in the supplement.

  1. How many participants in each analysis set have genotype data, and do the genotyped
     and ungenotyped differ in anything that matters (Table S14)?
  2. Does adding the ten genetic principal components, in the subset where they exist,
     change the principal results (Table S15)? The pre-registered rs174547 × microbiota
     interaction is reported here as well; it is the only genetic test in this manuscript.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from xnames import labels as taxa_labels
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
def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))

PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
SC  = pd.read_csv(f'panel_scores_{ST}.csv'); SC['subject_id'] = SC.subject_id.astype(str)
SC  = SC.set_index('subject_id')
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
from stage_v10 import master_stage
A['CKM5'] = master_stage(A, 'X4 MB825')
mb = A.set_index('subject_id'); S825 = list(A.subject_id)
S743 = [s for s in S825 if s in set(PHE.subject_id)]

PCS = pd.read_csv(f'{UP}/analysis/pheno_final_20260822_v25.csv')
PCS['subject_id'] = PCS.subject_id.astype(str)
PCS = PCS.set_index('subject_id')[[f'PC{i}' for i in range(1, 11)] + ['gbatch']]
GENO = set(PCS.index)                       # 有十個主成分（只在有代謝體的人身上算過）
# 20261030_v65：真正的「有基因型」要看基因檔本身，不是主成分檔
GTYPED = set(pd.read_csv(f'{UP}/analysis/pipeline/geno_panel500.csv.gz',
                         usecols=['subject_id']).subject_id.astype(str))
S585 = [s for s in S743 if s in GENO]
assert set(S585) == set(S743) & GTYPED, 'MB743 的有基因型與有主成分應該是同一群人'

# ---------------- Table S14: who has genotype ----------------
rows = []
for name, subj in [('Metabolomics (analysed for metabolite–outcome models)', list(PHE.subject_id)),
                   ('Microbiota profiling (MB825)', S825),
                   ('Microbiota profiling with metabolomics (MB743)', S743)]:
    g = [s for s in subj if s in GTYPED]
    pc = [s for s in subj if s in GENO]
    rows.append(dict(analysis_set=name, n=len(subj),
                     n_genotyped=len(g), pct_genotyped=round(100 * len(g) / len(subj), 1),
                     n_with_pcs=len(pc), pct_with_pcs=round(100 * len(pc) / len(subj), 1)))
GT = pd.DataFrame(rows)
add('n_genotyped_metabolome', int(GT.n_genotyped.iloc[0]), 'people', 'genotype data available')
add('n_genotyped_MB825', int(GT.n_genotyped.iloc[1]), 'people', 'genotype data available')
add('n_genotyped_MB743', int(GT.n_genotyped.iloc[2]), 'people', 'genotype data available')
add('pct_genotyped_MB825', float(GT.pct_genotyped.iloc[1]), '%', 'genotype data available')
add('n_with_pcs_MB825', int(GT.n_with_pcs.iloc[1]), 'people',
    'ten ancestry principal components available')
add('n_with_pcs_MB743', int(GT.n_with_pcs.iloc[2]), 'people',
    'ten ancestry principal components available')
add('n_trio_omics', len(S585), 'people',
    'microbiota + metabolomics + genotype; the genetic sensitivity subset')
add('n_genotyped_no_metabolome', len([s for s in S825 if s in GTYPED]) - len(S585), 'people',
    'genotyped but without metabolomics, so no ancestry PCs were computed')
add('n_microbiome_excluded_by_genotype', len(S825) - len([s for s in S825 if s in GTYPED]),
    'people', 'profiled participants a genotype requirement would have excluded')
# 這三個斷言把 issue 01 的三個數字釘住，之後再分岔會當場停下來
assert int(GT.n_genotyped.iloc[1]) == len(set(S825) & GTYPED)
assert int(GT.n_with_pcs.iloc[1]) == len(set(S825) & GENO) == len(S585)
assert len(set(S825) & GTYPED) - len(S585) == len([s for s in S825 if s not in set(S743)
                                                   and s in GTYPED])

# are the genotyped and ungenotyped comparable within MB825?
cmp_rows = []
# 20261030_v65：兩種分組都報。trio 是解讀 S15 真正要看的那一組。
GROUPS = [('genotyped', 'Genotype data available', GTYPED),
          ('trio', 'Microbiota + metabolomics + genotype', GENO)]
isg = np.array([s in GENO for s in S825])          # 下面的舊迴圈仍以 trio 為主要比較
for col, lab in [('age', 'Age, years'), ('BMI', 'Body-mass index, kg/m²'),
                 ('waist_cm', 'Waist circumference, cm'), ('SBP', 'Systolic BP, mmHg'),
                 ('lab_triglyceride', 'Triglycerides, mg/dL'), ('lab_HbA1c', 'HbA1c, %'),
                 ('lab_eGFR', 'eGFR, mL/min/1.73 m²'), ('CKM5', 'CKM stage')]:
    v = pd.to_numeric(mb[col], errors='coerce').reindex(S825).values
    a, b = v[isg & np.isfinite(v)], v[~isg & np.isfinite(v)]
    cmp_rows.append(dict(variable=lab,
                         genotyped=f'{a.mean():.1f} ({a.std():.1f})', n_genotyped=len(a),
                         not_genotyped=f'{b.mean():.1f} ({b.std():.1f})', n_not=len(b),
                         P=stats.mannwhitneyu(a, b).pvalue))
for col, lab in [('sex', 'Female sex'), ('C_DM', 'Diabetes mellitus')]:
    v = pd.to_numeric(mb[col], errors='coerce').reindex(S825).values
    a, b = v[isg & np.isfinite(v)], v[~isg & np.isfinite(v)]
    cmp_rows.append(dict(variable=lab,
                         genotyped=f'{int(a.sum())} ({100*a.mean():.1f}%)', n_genotyped=len(a),
                         not_genotyped=f'{int(b.sum())} ({100*b.mean():.1f}%)', n_not=len(b),
                         P=stats.chi2_contingency([[a.sum(), len(a) - a.sum()],
                                                   [b.sum(), len(b) - b.sum()]])[1]))
CMP = pd.DataFrame(cmp_rows)
pd.concat([GT.assign(variable='', genotyped='', not_genotyped=''), CMP],
          ignore_index=True).to_csv(f'TableS14_genotype_availability_{ST}.csv', index=False)
CMP.insert(0, 'comparison', 'trio-omics subset (n=585) vs the rest of MB825 (n=240)')
CMP.to_csv(f'TableS14b_genotyped_vs_not_{ST}.csv', index=False)
add('n_genotype_comparisons_p05', int((CMP.P < 0.05).sum()), 'of ' + str(len(CMP)),
    'trio-omics subset vs the rest of MB825')

# ---------------- Table S15: principal results with and without the ten PCs ----------------
GC = pd.read_csv(f'{HUB}/abundance_genus_counts.csv', index_col=0)
GX = GC.loc[A.sample_id]; KEEPG = ((GX > 0).mean(0) >= 0.10)
CL = np.log(GX.loc[:, KEEPG[KEEPG].index] + 0.5); CL = CL.sub(CL.mean(1), axis=0)
CL.index = A.subject_id.values; CL.columns = taxa_labels(list(CL.columns))
Z = (CL - CL.mean()) / CL.std()

ph = PHE.set_index('subject_id')
y1 = pd.Series(rint(ph.R1_PCae_lysoPC_z.values), index=ph.index)
age = pd.to_numeric(mb.age, errors='coerce'); sex = pd.to_numeric(mb.sex, errors='coerce')
bmi = pd.to_numeric(mb.BMI, errors='coerce')
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float)
tg = pd.Series(rint(np.log(pd.to_numeric(mb.lab_triglyceride, errors='coerce').reindex(S825))),
               index=S825)
ckm = pd.Series(rint(pd.to_numeric(mb.CKM5, errors='coerce').reindex(S825).values), index=S825)

def design(subj, with_pcs, extra=None):
    P = [np.ones(len(subj)), age.reindex(subj).values, sex.reindex(subj).values,
         bmi.reindex(subj).values, BAT.reindex(subj).values]
    if with_pcs:
        P += [PCS.reindex(subj)[[f'PC{i}' for i in range(1, 11)]].values,
              pd.get_dummies(PCS.reindex(subj).gbatch, drop_first=True).astype(float).values]
    if extra: P += [np.asarray(e, float).reshape(len(subj), -1) for e in extra]
    return np.column_stack(P)

def est(y, x, subj, with_pcs, extra=None, which=0):
    X = np.column_stack([np.asarray(x, float).reshape(len(subj), -1),
                         design(subj, with_pcs, extra)])
    yv = np.asarray(y, float); ok = np.isfinite(X).all(1) & np.isfinite(yv)
    X, yv = X[ok], yv[ok]
    nc = 1 if np.ndim(x) == 1 else np.shape(x)[1]
    keep = np.r_[np.ones(nc, bool), X[:, nc:].std(0) > 1e-10]; keep[nc] = True
    X = X[:, keep]
    b, se = hc3(X, yv)
    return b[which], se[which], 2 * stats.t.sf(abs(b[which] / se[which]),
                                               ok.sum() - X.shape[1]), int(ok.sum())

pan = SC.panel_oof; pri = SC.score_prior
rows = []
for lab, y, x, subj in [('Microbiota panel → ether-lipid ratio', y1.reindex(S743), pan.reindex(S743), S743),
                        ('Pre-specified score → ether-lipid ratio', y1.reindex(S743), pri.reindex(S743), S743),
                        ('Microbiota panel → log(triglycerides)', tg.reindex(S743), pan.reindex(S743), S743),
                        ('Microbiota panel → CKM stage', ckm.reindex(S743), pan.reindex(S743), S743)]:
    b0, s0, p0, n0 = est(y.values, x.values, subj, False)
    sg = [s for s in subj if s in GENO]
    b1, s1, p1, n1 = est(y.reindex(sg).values, x.reindex(sg).values, sg, False)
    b2, s2, p2, n2 = est(y.reindex(sg).values, x.reindex(sg).values, sg, True)
    rows.append(dict(result=lab,
                     beta_full=b0, p_full=p0, n_full=n0,
                     beta_genotyped_no_pcs=b1, p_genotyped_no_pcs=p1, n_genotyped=n1,
                     beta_genotyped_with_pcs=b2, p_genotyped_with_pcs=p2))
SENS = pd.DataFrame(rows); SENS.to_csv(f'TableS15_pc_sensitivity_{ST}.csv', index=False)
r0 = SENS.iloc[0]
add('sens_panel_R1_beta_full', r0.beta_full, '', ''); add('sens_panel_R1_p_full', r0.p_full, '', '')
add('sens_panel_R1_beta_pcs', r0.beta_genotyped_with_pcs, '', '')
add('sens_panel_R1_p_pcs', r0.p_genotyped_with_pcs, '', '')
add('sens_panel_R1_n_pcs', int(r0.n_genotyped), 'people', '')
add('max_abs_pc_shift', float(np.max(np.abs(SENS.beta_genotyped_with_pcs -
                                            SENS.beta_genotyped_no_pcs))), 'SD per SD',
    'largest change in any principal estimate from adding the ten PCs')

# ---------------- rs174547 × microbiota interaction (the only genetic test here) ----------------
G = pd.read_csv(f'{UP}/analysis/pipeline/geno_panel500.csv.gz', usecols=['subject_id', 'rs174547'])
G['subject_id'] = G.subject_id.astype(str); G = G.set_index('subject_id').rs174547
g = pd.to_numeric(G, errors='coerce'); g[g < 0] = np.nan
add('n_rs174547_called', int(g.reindex(S585).notna().sum()), 'people', 'of the genotyped MB743')
gz = (g - g.reindex(S585).mean()) / g.reindex(S585).std()
irows = []
for lab, x in [('Cross-validated panel', pan), ('Pre-specified score', pri)]:
    xs = x.reindex(S585).values; gs = gz.reindex(S585).values
    b, se, p, n = est(y1.reindex(S585).values, np.column_stack([xs * gs, xs, gs]),
                      S585, True, which=0)
    irows.append(dict(term=f'rs174547 × {lab} → ether-lipid ratio', beta=b, se=se, p=p, n=n))
    k = 'panel' if 'panel' in lab else 'prior'
    add(f'gxm_{k}_R1_beta', b, '', ''); add(f'gxm_{k}_R1_p', p, '', '')
    add(f'gxm_{k}_R1_n', n, 'people', '')
IX = pd.DataFrame(irows); IX.to_csv(f'TableS16_rs174547_interaction_{ST}.csv', index=False)

N = pd.read_csv(f'numbers_{ST}.csv')
N = N[~N.key.isin([f['key'] for f in F])]
pd.concat([N, pd.DataFrame(F)], ignore_index=True).to_csv(f'numbers_{ST}.csv', index=False)
print(f'{len(F)} numbers appended\n')
print(GT.to_string(index=False)); print()
print(CMP.to_string(index=False)); print()
print(SENS.round(4).to_string(index=False)); print()
print(IX.to_string(index=False))
