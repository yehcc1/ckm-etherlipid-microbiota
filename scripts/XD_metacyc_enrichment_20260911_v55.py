"""MetaCyc pathway enrichment on the PICRUSt2 output — 20260911_v55.

The analogue of Wu et al.'s Figure 4b, adapted to a continuous design. Their version asked
which Gemmiger-associated MetaCyc pathways differ between two clinical groups. Here there
are four exposures worth asking the same question of, so all four are run over the same
365 pathways with the same model and the same correction:

  A  the cross-validated microbiota panel   — the exposure that carries this paper's signal
  B  the plasma ether-lipid ratio           — the host readout the paper is about
  C  Gemmiger, centred log-ratio            — the taxon the earlier paper localised to
  D  CKM stage 3 or 4                       — the clinical contrast

Pathway abundances are compositional, so they are centred log-ratio transformed before
modelling, exactly as the genus table is. Every model is adjusted for age, sex, body-mass
index and sequencing batch with HC3 standard errors, and Benjamini-Hochberg is applied
within each exposure across all 365 pathways.

The analysis that decides whether any of this is interpretable is the last one: every hit
is re-estimated with the weighted nearest-sequenced taxon index in the model. If the hits
do not survive that, they are a reference-genome coverage artefact and not biology.

Finally, the 26 pathways Wu et al. report for Gemmiger are tested as a pre-specified set,
which is the right framing for a replication: the question is not whether we rediscover them
untargeted but whether their effects point the same way in an independent, larger sample.
"""
import warnings; warnings.filterwarnings('ignore')
import gzip
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from xnames import labels as taxa_labels
np.seterr(all='ignore')

UP = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'
PIC = f'{UP}/PICRUSt2 Galaxy/'
ST = '20260911_v55'
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

# ---------------- samples ----------------
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
from stage_v10 import master_stage
A = A.assign(CKM5=master_stage(A, 'XD MB825'))
mb = A.set_index('subject_id'); S = list(A.subject_id)
s2subj = dict(zip(A.sample_id, A.subject_id))

PA = pd.read_csv(PIC + 'Galaxy16-[PICRUSt2 Full pipeline on dataset 11 and 12_ '
                       'Pathway abundances].tabular', sep='\t', index_col=0)
PA = PA[[c for c in PA.columns if c in s2subj]]
PA.columns = [s2subj[c] for c in PA.columns]
PA = PA.T.reindex(S)                                   # participants x pathways
add('n_metacyc_pathways', PA.shape[1], 'pathways', 'PICRUSt2 MetaCyc output')
add('n_picrust_samples', PA.shape[0], 'people', '')
PC = np.log(PA + 0.5); PC = PC.sub(PC.mean(1), axis=0)  # CLR across pathways
PZ = (PC - PC.mean()) / PC.std()

NS = pd.read_csv(PIC + 'Galaxy21-[EC_weighted_nsti].tabular', sep='\t')
NS['subject_id'] = NS['sample'].map(s2subj)
nsti = NS.dropna(subset=['subject_id']).set_index('subject_id').weighted_NSTI.reindex(S)
add('nsti_median', round(float(nsti.median()), 3), '', 'weighted NSTI, 825 samples')
add('nsti_pct_above_015', round(100 * float((nsti > 0.15).mean()), 1), '%', '')

# ---------------- exposures ----------------
GC = pd.read_csv(f'{HUB}/abundance_genus_counts.csv', index_col=0)
GX = GC.loc[A.sample_id]; GX = GX.loc[:, (GX > 0).mean(0) >= 0.10]
CL = np.log(GX + 0.5); CL = CL.sub(CL.mean(1), axis=0)
CL.index = A.subject_id.values; CL.columns = taxa_labels(list(GX.columns))
GZ = ((CL - CL.mean()) / CL.std())

PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
ph = PHE.set_index('subject_id')
r1 = pd.Series(rint(ph.R1_PCae_lysoPC_z.values), index=ph.index).reindex(S)
SC = pd.read_csv('panel_scores_20260911_v54.csv'); SC['subject_id'] = SC.subject_id.astype(str)
SC = SC.set_index('subject_id')
panel = SC.panel_oof.reindex(S)
gemmiger = GZ['Gemmiger'].reindex(S)
ckm34 = pd.Series((pd.to_numeric(mb.CKM5, errors='coerce').reindex(S).values >= 3).astype(float),
                  index=S)

age = pd.to_numeric(mb.age, errors='coerce').reindex(S)
sex = pd.to_numeric(mb.sex, errors='coerce').reindex(S)
bmi = pd.to_numeric(mb.BMI, errors='coerce').reindex(S)
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float).reindex(S)

def scan(x, extra=None):
    """every pathway against one exposure, adjusted; pathway is the outcome."""
    cov = [np.ones(len(S)), age.values, sex.values, bmi.values, BAT.values]
    if extra is not None: cov.append(np.asarray(extra, float))
    Cv = np.column_stack(cov)
    xv = np.asarray(x, float); out = []
    for p_ in PZ.columns:
        y = PZ[p_].values
        X = np.column_stack([xv, Cv])
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        Xo = X[ok]; keep = np.r_[True, Xo[:, 1:].std(0) > 1e-10]; keep[1] = True
        Xo = Xo[:, keep]
        b, se = hc3(Xo, y[ok])
        out.append(dict(pathway=p_, beta=b[0], se=se[0],
                        p=2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - Xo.shape[1]),
                        n=int(ok.sum())))
    D = pd.DataFrame(out); D['q'] = bh(D.p.values); return D

EXP = {'Microbiota panel': panel, 'Ether-lipid ratio': r1,
       'Gemmiger': gemmiger, 'CKM stage 3 or 4': ckm34}
res = []
for nm, x in EXP.items():
    D = scan(x); D.insert(0, 'exposure', nm); res.append(D)
    add(f'metacyc_{nm.split()[0].lower()}_n_q05', int((D.q < 0.05).sum()), 'pathways', '')
    add(f'metacyc_{nm.split()[0].lower()}_n_p05', int((D.p < 0.05).sum()), 'pathways', '')
R = pd.concat(res, ignore_index=True)

# ---------------- the test that decides interpretability ----------------
nz = rint(nsti.values)
res2 = []
for nm, x in EXP.items():
    D = scan(x, extra=nz); D.insert(0, 'exposure', nm); res2.append(D)
R2 = pd.concat(res2, ignore_index=True)
M = R.merge(R2[['exposure', 'pathway', 'beta', 'p', 'q']], on=['exposure', 'pathway'],
            suffixes=('', '_nsti_adj'))

# ---------------- pathway names, from the PICRUSt2 description mapfile ----------------
# 原始 PICRUSt2 mapfile 不在專案資料夾裡；名稱改由已發表的 Table S17 取得。
# 名稱只用於顯示與 Wu 等人的集合比對（該比對本來就限定在本次測到的 365 條內），
# 統計完全不受影響。
NAME = dict(l.split('\t', 1) for l in gzip.open(
    'metacyc_names_from_S17.txt.gz', 'rt', encoding='utf-8', errors='replace'
).read().rstrip('\n').split('\n'))
missing = sorted(set(M.pathway) - set(NAME))
assert not missing, f'pathway IDs with no MetaCyc name: {missing[:5]}'
M.insert(2, 'name', M.pathway.map(NAME))
M.to_csv(f'TableS17_metacyc_enrichment_{ST}.csv', index=False)

print(f'MetaCyc pathways tested: {PZ.shape[1]}   participants: {PZ.shape[0]}')
print(f'weighted NSTI: median {nsti.median():.3f}, {(nsti>0.15).mean()*100:.1f}% above 0.15\n')
summ = []
for nm in EXP:
    d = M[M.exposure == nm]
    summ.append(dict(exposure=nm,
                     p05=int((d.p < 0.05).sum()), q05=int((d.q < 0.05).sum()),
                     q05_after_NSTI=int(((d.q < 0.05) & (d.q_nsti_adj < 0.05)).sum()),
                     median_abs_beta_shift=round(float(
                         (d.beta - d.beta_nsti_adj).abs().median()), 4)))
SUM = pd.DataFrame(summ); print(SUM.to_string(index=False))
for nm in EXP:
    sel = M.exposure == nm
    # 20261030_v65：三種數量分開存，圖與正文才不會把交集講成校正模型的發現數
    add(f'metacyc_{nm.split()[0].lower()}_n_q05_base',
        int((sel & (M.q < 0.05)).sum()), 'pathways', 'q<0.05 in the primary model')
    add(f'metacyc_{nm.split()[0].lower()}_n_q05_adjmodel',
        int((sel & (M.q_nsti_adj < 0.05)).sum()), 'pathways',
        'q<0.05 in the NSTI-adjusted model, irrespective of the primary model')
    add(f'metacyc_{nm.split()[0].lower()}_survive_nsti',
        int((sel & (M.q < 0.05) & (M.q_nsti_adj < 0.05)).sum()), 'pathways',
        'q<0.05 in BOTH models (the intersection)')

print('\nTop pathways by exposure (q < 0.05, up to 12 each):')
for nm in EXP:
    d = M[(M.exposure == nm) & (M.q < 0.05)].sort_values('p')
    print(f'\n--- {nm}  ({len(d)} at q < 0.05) ---')
    if len(d):
        print(d.head(12)[['pathway', 'beta', 'p', 'q', 'q_nsti_adj']].to_string(index=False))

# how much of the pathway signal is simply NSTI?
nrows = []
for p_ in PZ.columns:
    y = PZ[p_].values
    X = np.column_stack([nz, np.ones(len(S)), age.values, sex.values, bmi.values, BAT.values])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    Xo = X[ok]; keep = np.r_[True, Xo[:, 1:].std(0) > 1e-10]; keep[1] = True
    b, se = hc3(Xo[:, keep], y[ok])
    nrows.append(dict(pathway=p_, beta=b[0], se=se[0],
                      p=2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - keep.sum())))
NR = pd.DataFrame(nrows); NR['q'] = bh(NR.p.values)
NR.insert(1, 'name', NR.pathway.map(NAME))
NR.to_csv(f'TableS18_metacyc_vs_nsti_{ST}.csv', index=False)
add('metacyc_nsti_n_q05', int((NR.q < 0.05).sum()), 'pathways', 'pathways predicted by NSTI alone')
print(f'\nPathways associated with NSTI itself: {(NR.q<0.05).sum()} of {len(NR)} at q < 0.05')
print(f'Correlation of NSTI with the microbiota panel: '
      f'r = {np.corrcoef(nz[np.isfinite(nz)&np.isfinite(panel.values)], panel.values[np.isfinite(nz)&np.isfinite(panel.values)])[0,1]:+.3f}')

# ---------------- targeted replication of the earlier paper's Gemmiger pathway set ----
# 20261030_v65：改用 Wu et al. Figure 4b 上的 25 條，以 MetaCyc ID 精確指定。
# 原本用名稱子字串比對，「galactose degradation I」會把 LACTOSECAT-PWY
# （lactose and galactose degradation I）一起撈進來，而那一條圖上沒有，26 就是這樣來的。
# 圖上四組：胺基酸 9、醣類 8、核苷酸 4、其他 4（含 MEP pathway I 與 II 兩條）。
WU_IDS = {
    'PWY-6151': 'S-adenosyl-L-methionine cycle I',
    'ILEUSYN-PWY': 'L-isoleucine biosynthesis I (from threonine)',
    'PWY-5101': 'L-isoleucine biosynthesis II',
    'PWY-5103': 'L-isoleucine biosynthesis III',
    'BRANCHED-CHAIN-AA-SYN-PWY': 'superpathway of branched amino acid biosynthesis',
    'PWY-3001': 'superpathway of L-isoleucine biosynthesis I',
    'VALSYN-PWY': 'L-valine biosynthesis',
    'PWY-5104': 'L-isoleucine biosynthesis IV',
    'HISTSYN-PWY': 'L-histidine biosynthesis',
    'PWY-621': 'sucrose degradation III (sucrose invertase)',
    'PWY-6317': 'galactose degradation I (Leloir pathway)',
    'PWY-6737': 'starch degradation V',
    'GLYCOGENSYNTH-PWY': 'glycogen biosynthesis I (from ADP-D-Glucose)',
    'PWY-5100': 'pyruvate fermentation to acetate and lactate II',
    'PWY-7111': 'pyruvate fermentation to isobutanol (engineered)',
    'PWY-5384': 'sucrose degradation IV (sucrose phosphorylase)',
    'ANAGLYCOLYSIS-PWY': 'glycolysis III (from glucose)',
    'PWY0-1296': 'purine ribonucleosides degradation',
    'PWY-6609': 'adenine and adenosine salvage III',
    'PWY0-1297': 'superpathway of purine deoxyribonucleosides degradation',
    'DTDPRHAMSYN-PWY': 'dTDP-L-rhamnose biosynthesis I',
    'NONMEVIPP-PWY': 'methylerythritol phosphate pathway I',
    'PWY-7560': 'methylerythritol phosphate pathway II',
    'PWY-5121': 'superpathway of geranylgeranyl diphosphate biosynthesis II (via MEP)',
    'P161-PWY': 'acetylene degradation',
}
assert len(WU_IDS) == 25, len(WU_IDS)
avail = set(M.pathway)
missing = [k for k in WU_IDS if k not in avail]
assert not missing, f'Wu 的路徑不在我們的 365 條裡: {missing}'
# 每一個 ID 的名稱必須與圖上的標籤相符，否則是抄錯 ID
for k, lab in WU_IDS.items():
    nm = str(NAME.get(k, '')).lower()
    key = lab.split(' (')[0].lower()
    assert key in nm, f'{k}: 我們的名稱 {nm!r} 與圖上的 {lab!r} 不符'
ids = sorted(WU_IDS)
REP = M[(M.exposure == 'Gemmiger') & (M.pathway.isin(ids))].copy().sort_values('p')
REP.to_csv(f'TableS19_replication_prior_paper_{ST}.csv', index=False)
same = int((REP.beta > 0).sum()); tot = len(REP)
binom = float(stats.binomtest(same, tot, 0.5).pvalue)
add('rep_n_pathways', tot, 'pathways', "Wu et al. Figure 4b Gemmiger set, 25 MetaCyc IDs")
add('rep_n_same_direction', same, 'pathways', '')
add('rep_binom_p', binom, '', 'direction agreement, two-sided binomial')
add('rep_n_p05', int((REP.p < 0.05).sum()), 'pathways', '')
add('rep_n_survive_nsti', int((REP.q_nsti_adj < 0.05).sum()), 'pathways', '')
print(f'\nTargeted replication of the earlier paper\'s Gemmiger pathway set:')
print(f'  {tot} of their pathways are present here; {same} point the same way '
      f'(binomial P = {binom:.2g}); {(REP.p<0.05).sum()} nominally significant; '
      f'{(REP.q_nsti_adj<0.05).sum()} survive NSTI adjustment')

# the pathway that matters mechanistically
for exp_ in ['Microbiota panel', 'Ether-lipid ratio']:
    r_ = M[(M.exposure == exp_) & (M.pathway == 'PWY-5971')].iloc[0]
    k = 'panel' if exp_.startswith('Micro') else 'r1'
    add(f'pwy5971_{k}_beta', r_.beta, '', ''); add(f'pwy5971_{k}_q', r_.q, '', '')
    add(f'pwy5971_{k}_q_nsti', r_.q_nsti_adj, '', '')
    print(f'  PWY-5971 ({r_["name"]}) vs {exp_}: '
          f'beta = {r_.beta:+.3f}, q = {r_.q:.2g}, q after NSTI = {r_.q_nsti_adj:.2g}')

pd.DataFrame(F).to_csv(f'numbers_metacyc_{ST}.csv', index=False)
