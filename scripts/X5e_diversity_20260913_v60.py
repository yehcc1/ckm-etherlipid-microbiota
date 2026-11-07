"""Alpha and beta diversity in the 825 profiled participants — 20260912_v60.

Nothing here was in the manuscript before. It answers the question every microbiome reviewer
asks: does the community as a whole differ across CKM stage, and does it differ with the
exposures this paper uses?

Alpha diversity is taken at ASV level from the cohort's own diversity table (observed ASVs,
Shannon, Pielou evenness, read depth). Beta diversity is computed at genus level, both as
Bray-Curtis on relative abundance over every genus and as Aitchison (Euclidean on the centred
log-ratio of the 10%-prevalence genera), i.e. the same matrix the rest of the paper models.

PERMANOVA is implemented here rather than imported: SS = tr(HG) with G the Gower-centred
matrix of squared distances and H the hat matrix of the design, tested by free permutation of
the sample labels (999), terms entered sequentially — the default behaviour of adonis2.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'
ST = '20260913_v60'
RNG = np.random.default_rng(20260912)
F = []
def add(k, v, u='', s=''): F.append(dict(key=k, value=v, unit=u, source=s))

# ------------------------------------------------------------------ the 825 set
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = (A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
        (~A0.subject_id.str.startswith('J'))]
     .drop_duplicates('subject_id').copy())
from stage_v10 import master_stage
A['CKM5'] = master_stage(A, 'X5e MB825')
A = A.set_index('subject_id')
n825 = len(A)

PS = pd.read_csv('panel_scores_20260911_v54.csv').set_index('subject_id')
PS.index = PS.index.astype(str)
PHE = pd.read_csv('pheno_all_20260911_v54.csv')
PHE['subject_id'] = PHE.subject_id.astype(str)
rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)),
        np.full(len(np.asarray(v, float)), np.nan))
PHE['R1_int'] = rint(PHE['R1_PCae_lysoPC_z'].values)
R1 = PHE.set_index('subject_id').R1_int

# ------------------------------------------------------------------ alpha
AL = pd.read_csv(f'{HUB}/alpha_diversity.csv')
AL = AL.set_index('sample_id').reindex(A.sample_id.values)
AL.index = A.index
assert AL.shannon.notna().all(), 'alpha diversity missing for some analysis samples'
ALPHA = {'Observed ASVs': AL.n_asv, 'Shannon': AL.shannon,
         'Pielou evenness': AL.evenness, 'Read depth': AL.total_reads}
add('alpha_reads_median', float(AL.total_reads.median()), 'reads', 'hub alpha table')
add('alpha_reads_min', float(AL.total_reads.min()), 'reads', '')
add('alpha_shannon_median', round(float(AL.shannon.median()), 2), '', '')
add('alpha_nasv_median', float(AL.n_asv.median()), 'ASVs', '')

rows = []
for nm, v in ALPHA.items():
    g = [v[A.CKM5 == s].dropna().values for s in range(5)]
    H, p = stats.kruskal(*[x for x in g if len(x) > 2])
    rows.append(dict(metric=nm, test='Kruskal-Wallis across CKM stages 0-4',
                     stat=H, p=p, n=int(v.notna().sum()),
                     detail='; '.join(f'stage {s}: median {np.median(x):.3g} (n={len(x)})'
                                      for s, x in enumerate(g))))
    for xn, x in [('CV panel', PS.panel_oof.reindex(A.index)),
                  ('Pre-specified score', PS.score_prior.reindex(A.index)),
                  ('Ether-lipid ratio', R1.reindex(A.index))]:
        ok = v.notna() & x.notna()
        r, p2 = stats.spearmanr(v[ok], x[ok])
        rows.append(dict(metric=nm, test=f'Spearman vs {xn}', stat=r, p=p2, n=int(ok.sum()),
                         detail=''))
ALPHA_T = pd.DataFrame(rows)

# adjusted models: alpha ~ exposure + age + sex + BMI + batch
def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))
BAT = pd.get_dummies(A.batch, drop_first=True).astype(float)
adj = []
for nm, v in ALPHA.items():
    if nm == 'Read depth': continue
    for xn, x in [('CV panel', PS.panel_oof.reindex(A.index)),
                  ('Pre-specified score', PS.score_prior.reindex(A.index)),
                  ('Ether-lipid ratio', R1.reindex(A.index))]:
        d = pd.DataFrame({'y': rint(v.values), 'x': x.values,
                          'age': pd.to_numeric(A.age, errors='coerce').values,
                          'sex': pd.to_numeric(A.sex, errors='coerce').values,
                          'bmi': pd.to_numeric(A.BMI, errors='coerce').values},
                         index=A.index).join(BAT).dropna()
        X = np.column_stack([np.ones(len(d)), stats.zscore(d.x.values), d[['age', 'sex', 'bmi']].values,
                             d[BAT.columns].values])
        b, se = hc3(X, d.y.values)
        adj.append(dict(metric=nm, exposure=xn, beta=b[1], se=se[1],
                        p=2 * stats.norm.sf(abs(b[1] / se[1])), n=len(d)))
ALPHA_ADJ = pd.DataFrame(adj)

# ------------------------------------------------------------------ beta
GC = pd.read_csv(f'{HUB}/abundance_genus_counts.csv', index_col=0)
GX = GC.loc[A.sample_id]
REL = GX.div(GX.sum(1), axis=0).values
prev = (GX > 0).mean(0); KEEPG = prev[prev >= 0.10].index
CL = np.log(GX[KEEPG] + 0.5); CL = CL.sub(CL.mean(1), axis=0).values

def braycurtis(Rm):
    s = Rm.sum(1)
    D = np.zeros((len(Rm), len(Rm)))
    for i in range(len(Rm)):
        mn = np.minimum(Rm[i], Rm).sum(1)
        D[i] = 1 - 2 * mn / (s[i] + s)
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2

def euclid(Xm):
    q = (Xm ** 2).sum(1)
    D2 = np.clip(q[:, None] + q[None, :] - 2 * Xm @ Xm.T, 0, None)
    return np.sqrt(D2)

DIST = {'Bray-Curtis (all genera)': braycurtis(REL),
        'Aitchison (CLR, prevalent genera)': euclid(CL)}

def gower(D):
    n = len(D); J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ (D ** 2) @ J

def hat(X):
    return X @ np.linalg.pinv(X.T @ X) @ X.T

def permanova(D, terms, nperm=999, rng=RNG):
    """sequential (type I) PERMANOVA; terms is a list of (name, column block)."""
    n = len(D); G = gower(D); total = np.trace(G)
    blocks = [np.ones((n, 1))] + [np.asarray(b, float).reshape(n, -1) for _, b in terms]
    def ss_seq(order):
        out, prev, X = [], 0.0, blocks[0]
        for b in blocks[1:]:
            X = np.column_stack([X, b[order]])
            cur = float(np.sum(hat(X) * G))
            out.append(cur - prev); prev = cur
        return np.array(out), prev
    idn = np.arange(n)
    ss, model = ss_seq(idn)
    dfs = np.array([b.shape[1] for b in blocks[1:]], float)
    res = total - model; dfres = n - 1 - dfs.sum()
    Fobs = (ss / dfs) / (res / dfres)
    cnt = np.zeros(len(ss))
    for _ in range(nperm):
        o = rng.permutation(n)
        s2, m2 = ss_seq(o)
        f2 = (s2 / dfs) / ((total - m2) / dfres)
        cnt += (f2 >= Fobs - 1e-12)
    return pd.DataFrame(dict(term=[t for t, _ in terms], R2=ss / total, F=Fobs,
                             p=(cnt + 1) / (nperm + 1), df=dfs.astype(int)))

age = pd.to_numeric(A.age, errors='coerce')
sex = pd.to_numeric(A.sex, errors='coerce')
bmi = pd.to_numeric(A.BMI, errors='coerce')
stage = pd.get_dummies(A.CKM5, drop_first=True).astype(float)
panel = PS.panel_oof.reindex(A.index)
r1 = R1.reindex(A.index)

BETA, PCO = [], {}
for dn, D in DIST.items():
    G = gower(D); w, V = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    pc = V[:, :3] * np.sqrt(np.clip(w[:3], 0, None))
    PCO[dn] = dict(pc=pc, var=np.clip(w, 0, None)[:3] / np.clip(w, 0, None).sum())
    # every variable on its own
    for nm, x in [('16S batch', pd.get_dummies(A.batch, drop_first=True).astype(float)),
                  ('Age', age), ('Sex', sex), ('Body-mass index', bmi),
                  ('CKM stage', stage), ('Microbiota panel', panel),
                  ('Ether-lipid ratio', r1)]:
        xv = pd.DataFrame(x).astype(float)
        ok = xv.notna().all(1).values
        t = permanova(D[np.ix_(ok, ok)], [(nm, xv.values[ok])])
        t['distance'] = dn; t['model'] = 'univariable'; t['n'] = int(ok.sum())
        BETA.append(t)
    # the adjusted model, complete cases
    dd = pd.concat([age.rename('age'), sex.rename('sex'), bmi.rename('bmi'),
                    panel.rename('panel'), r1.rename('r1')], axis=1)
    ok = dd.notna().all(1).values
    terms = [('16S batch', pd.get_dummies(A.batch, drop_first=True).astype(float).values[ok]),
             ('Age', age.values[ok, None]), ('Sex', sex.values[ok, None]),
             ('Body-mass index', bmi.values[ok, None]), ('CKM stage', stage.values[ok]),
             ('Microbiota panel', panel.values[ok, None]), ('Ether-lipid ratio', r1.values[ok, None])]
    t = permanova(D[np.ix_(ok, ok)], terms)
    t['distance'] = dn; t['model'] = 'sequential, adjusted'; t['n'] = int(ok.sum())
    BETA.append(t)
BETA = pd.concat(BETA, ignore_index=True)[['distance', 'model', 'term', 'df', 'R2', 'F', 'p', 'n']]

ALPHA_T.to_csv(f'TableS26_alpha_diversity_{ST}.csv', index=False)
ALPHA_ADJ.to_csv(f'TableS26b_alpha_adjusted_{ST}.csv', index=False)
BETA.to_csv(f'TableS27_permanova_{ST}.csv', index=False)
np.savez(f'beta_pcoa_{ST}.npz',
         **{f'{k}_pc': v['pc'] for k, v in PCO.items()},
         **{f'{k}_var': v['var'] for k, v in PCO.items()},
         stage=A.CKM5.values, batch=A.batch.astype(str).values,
         panel=panel.values, r1=r1.values)

for _, r in ALPHA_T.iterrows():
    add(f'alpha_{r.metric.split()[0].lower()}_{r.test.split()[0].lower()}_p', float(r.p))
for _, r in BETA.iterrows():
    tag = r.term.lower().replace(' ', '').replace('-', '')
    dtag = 'bc' if r.distance.startswith('Bray') else 'ait'
    mtag = 'uni' if r.model == 'univariable' else 'adj'
    add(f'beta_{dtag}_{mtag}_{tag}_r2', float(r.R2)); add(f'beta_{dtag}_{mtag}_{tag}_p', float(r.p))
pd.DataFrame(F).to_csv(f'numbers_diversity_{ST}.csv', index=False)

pd.set_option('display.width', 200)
print(f'n = {n825}; read depth median {AL.total_reads.median():,.0f}, min {AL.total_reads.min():,.0f}\n')
print('ALPHA DIVERSITY')
print(ALPHA_T[['metric', 'test', 'stat', 'p', 'n']].to_string(index=False, float_format='%.4g'))
print('\nALPHA, adjusted for age, sex, BMI, batch (SD per SD)')
print(ALPHA_ADJ.to_string(index=False, float_format='%.4g'))
print('\nPERMANOVA')
print(BETA.to_string(index=False, float_format='%.4g'))
print(f'\nPCoA variance explained: ' +
      '; '.join(f'{k}: ' + ', '.join(f'{100*x:.1f}%' for x in v['var']) for k, v in PCO.items()))
