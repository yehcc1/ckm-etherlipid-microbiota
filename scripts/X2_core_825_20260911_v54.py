"""Core recomputation with the genotype gate removed — 20260911_v54.

One script, one specification, one numbers file.

What changed from v53
---------------------
The analysis set was previously metabolome ∩ genotype ∩ 16S = 585, because every model
carried ten genetic principal components. Genotype is no longer an entry requirement:

  MB825   every participant with a preferred 16S sample within 180 days of a clinical
          visit (n = 825). Used for analyses that need only microbiota and clinical data.
  MB743   MB825 ∩ targeted metabolomics (n = 743). Used for every analysis with a
          metabolite outcome or mediator. 82 of the 825 have no metabolomics at all.
  METAB   every participant with metabolomics and a clinical visit (n = 3,679). Used for
          the metabolite → clinical outcome models, which no longer need genotype either.

Covariates for microbiome models are age, sex, body-mass index at the stool-matched visit
and 16S sequencing batch. The ten genetic principal components and genotyping batch are
gone; the genotyped subset is re-analysed with them in X4 as a sensitivity analysis.

rank-INT is applied within the largest sample in which the variable is defined (metabolite
ratios across the 3,688 with metabolomics; clinical outcomes across METAB), not within
each subset, so that subsets remain comparable.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import KFold
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

def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty(len(p)); run = 1.0
    for i, idx in enumerate(o[::-1]):
        run = min(run, p[idx] * len(p) / (len(p) - i)); q[idx] = run
    return q

# ======================= data =======================
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
RAT = ['R1_PCae_lysoPC_z', 'R2_PCae362_361_z', 'R3_SM_over_PC_z', 'R4_SM240_160_z',
       'R5_lysoPC204_PCaa384_z', 'R6_PCaa384_383_z', 'R7_C16_C0_z', 'R8_C2_C0_z']
RLAB = dict(zip(RAT, ['R1 PC ae / lysoPC', 'R2 PC ae C36:2/C36:1', 'R3 SM / PC',
                      'R4 SM C24:0/C16:0', 'R5 lysoPC C20:4 / PC aa C38:4',
                      'R6 PC aa C38:4/C38:3', 'R7 C16/C0', 'R8 C2/C0']))
for r in RAT: PHE[r + '_int'] = rint(PHE[r].values)          # rank-INT over all 3,688

A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id').copy()
C = pd.read_csv(f'{UP}/CKM_clinical_for_microbiome_20260819_V6.csv', low_memory=False)
C['subject_id'] = C.subject_id.astype(str)
C = C.sort_values('visit_no').drop_duplicates('subject_id').copy()
RNA = set(pd.read_csv(f'{UP}/analysis/rnaseq_analysis_ids_20260823_v38.csv').subject_id.astype(str))

# ---- CKM stage now comes from the master table (大表 V10, CKM_stage_v3) ----------
# 使用者裁示：大表上怎麼寫他是第幾期他就是。分期取「各人採血當次那一列」的
# CKM_stage_v3；V10 以各人自己的 data exam 當判定時點，第四期來自抽血日之前的
# 診斷史（CGRD 住院≥1／急診≥1／門診≥2 次且間隔≥30 天）。大表沒有該訪次判定時
# （無 exam 日期或嘉義），退回舊式 OFF(CKM_stage, stage3_criteria_met)，並印出筆數。
OFF = lambda st, cm: np.where(st == 4, 4, np.where((cm >= 1) & (np.isin(st, [1, 2])), 3, st))
V10 = pd.read_csv('stage_v10_lookup_20261010_v64.csv')
V10['subject_id'] = V10.subject_id.astype(str)
V10['exam'] = pd.to_datetime(V10.exam, errors='coerce')
V10 = V10[['subject_id', 'exam', 'CKM_stage_v3']].dropna(subset=['exam'])

def stage_from_master(df, tag):
    old = OFF(df.CKM_stage.values, df.stage3_criteria_met.values).astype(float)
    k = df[['subject_id']].copy()
    k['exam'] = pd.to_datetime(df.exam, errors='coerce')
    new = k.merge(V10, on=['subject_id', 'exam'], how='left').CKM_stage_v3.values
    hit = np.isfinite(new)
    out = np.where(hit, new, old)
    chg = int(np.sum(hit & (new != old)))
    print(f'  {tag}: 大表 V10 對到 {hit.sum()}/{len(df)} '
          f'（{100 * hit.mean():.1f}%），與舊分期不同 {chg} 人；'
          f'其餘 {int((~hit).sum())} 人沿用 OFF(CKM_stage, stage3_criteria_met)')
    return out

print('CKM stage 來源：大表 V10 CKM_stage_v3（採血當次）')
A['CKM5'] = stage_from_master(A, 'MB825')
C['CKM5'] = stage_from_master(C, 'clinical')
add('stage_source', 'master table V10 CKM_stage_v3 at the blood-draw visit', '', '大表')
MED = [c for c in A.columns if c.startswith('med_')]
KEY = [c for c in ['med_Statin', 'med_Metformin', 'med_PPARalpha_Fibrate', 'med_SGLT2i',
                   'med_GLP1_RA', 'med_Insulin', 'med_Ezetimibe', 'med_PPARgamma_TZD']
       if c in A.columns]

S825 = list(A.subject_id)
S743 = [s for s in S825 if s in set(PHE.subject_id)]
A = A.set_index('subject_id').loc[S825].reset_index()
add('n_microbiome_all', len(S825), 'people', 'analysis_table_v7, preferred sample, <=180 d')
add('n_microbiome', len(S743), 'people', 'MB825 with targeted metabolomics')
add('n_microbiome_no_metabolome', len(S825) - len(S743), 'people', '')
add('n_metabolome_total', len(PHE), 'people', 'p180_all')
add('n_rnaseq_overlap', len(set(S743) & RNA), 'people', 'rnaseq ids v38')
add('mb_delta_days_median', float(A.delta_days.abs().median()), 'days', '')
add('n_medication_classes', len(MED), 'classes', '')

# ======================= microbiota matrix =======================
GC = pd.read_csv(f'{HUB}/abundance_genus_counts.csv', index_col=0)
GX = GC.loc[A.sample_id]
prev = (GX > 0).mean(0)
KEEPG = prev[prev >= 0.10].index
readfrac = GX[KEEPG].sum().sum() / GX.sum().sum()
CL = np.log(GX[KEEPG] + 0.5); CL = CL.sub(CL.mean(1), axis=0)
CL.index = A.subject_id.values
from xnames import labels as taxa_labels
GEN = taxa_labels(list(CL.columns))
CL.columns = GEN
assert len(set(GEN)) == len(GEN), 'genus labels are not unique'
add('n_genera_total', int(GC.shape[1]), 'genera', 'genus count table')
add('n_genera_tested', int(len(KEEPG)), 'genera', '10% prevalence filter, MB825')
add('pct_reads_retained', round(100 * readfrac, 1), '%', '')

# Firmicutes / Bacteroidota, from the same genus table
fc = [c for c in GC.columns if re.search(r'Firmicutes|Bacillota', c, re.I)]
bc = [c for c in GC.columns if re.search(r'Bacteroidota|Bacteroidetes', c, re.I)]
FB = pd.Series(np.log((GC.loc[A.sample_id, fc].sum(1) + 0.5) /
                      (GC.loc[A.sample_id, bc].sum(1) + 0.5)).values, index=A.subject_id.values)

Z = (CL - CL.mean()) / CL.std()                      # standardised CLR, MB825

# ======================= design matrices =======================
mb = A.set_index('subject_id')
age_mb = pd.to_numeric(mb.age, errors='coerce')
sex_mb = pd.to_numeric(mb.sex, errors='coerce')
bmi_mb = pd.to_numeric(mb.BMI, errors='coerce')
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float)

def cov_for(subjects, with_bmi=True, extra=None):
    idx = list(subjects)
    parts = [np.ones(len(idx)), age_mb.reindex(idx).values, sex_mb.reindex(idx).values]
    if with_bmi: parts.append(bmi_mb.reindex(idx).values)
    Mx = np.column_stack(parts + [BAT.reindex(idx).values])
    if extra is not None: Mx = np.column_stack([Mx] + [np.asarray(e, float) for e in extra])
    return Mx

def fit(y, x, subjects, with_bmi=True, extra=None, binary=False, which=0):
    idx = list(subjects)
    X0 = np.column_stack([np.asarray(x, float)] if np.ndim(x) == 1 else x)
    Cv = cov_for(idx, with_bmi, extra)
    X = np.column_stack([X0, Cv]); yv = np.asarray(y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(yv)
    X, yv = X[ok], yv[ok]
    keep = np.r_[np.ones(X0.shape[1], bool), X[:, X0.shape[1]:].std(0) > 1e-10]
    keep[X0.shape[1]] = True
    X = X[:, keep]
    if binary:
        r = sm.GLM(yv, X, family=sm.families.Binomial()).fit(cov_type='HC3')
        return r.params[which], r.bse[which], r.pvalues[which], int(ok.sum())
    b, se = hc3(X, yv)
    return b[which], se[which], 2 * stats.t.sf(abs(b[which] / se[which]),
                                               ok.sum() - X.shape[1]), int(ok.sum())

# ======================= phenotypes on the microbiome sets =======================
ph = PHE.set_index('subject_id')
Y743 = {r: ph[r + '_int'].reindex(S743).values for r in RAT}
Y825 = {r: ph[r + '_int'].reindex(S825).values for r in RAT}

# ======================= exposures =======================
def select_stats(train_idx, y, subjects):
    """每一屬在訓練列上的校正關聯：回傳 beta、t、p。

    共變數取自 cov_for，含定序批次；p 用 t 分布（df = n - k）。
    20261030_v65：原本只回傳 sign 權重，X3 因此自己又算了一次而且少了批次，
    兩邊的 >=3 fold 名單才會一個 19 屬一個 18 屬。改成把統計一起交出去。"""
    idx = [subjects[i] for i in train_idx]
    Cv = cov_for(idx); yv = np.asarray(y, float)[train_idx]
    nb = Z.shape[1]
    beta = np.full(nb, np.nan); tt = np.full(nb, np.nan); pp = np.full(nb, np.nan)
    for j, g in enumerate(Z.columns):
        v = Z[g].reindex(idx).values
        X = np.column_stack([v, Cv]); ok = np.isfinite(X).all(1) & np.isfinite(yv)
        if ok.sum() < 30: continue
        Xi = np.linalg.pinv(X[ok].T @ X[ok]); b = Xi @ (X[ok].T @ yv[ok])
        e = yv[ok] - X[ok] @ b
        se = np.sqrt((e @ e / (ok.sum() - X.shape[1])) * Xi[0, 0])
        beta[j] = b[0]; tt[j] = b[0] / se
        pp[j] = 2 * stats.t.sf(abs(tt[j]), ok.sum() - X.shape[1])
    return beta, tt, pp


def select(train_idx, y, subjects):
    """P<0.01 的 sign 權重，直接由 select_stats 導出。"""
    beta, _, pp = select_stats(train_idx, y, subjects)
    w = np.zeros(Z.shape[1])
    hit = np.isfinite(pp) & (pp < 0.01)
    w[hit] = np.sign(beta[hit])
    return w

y1_743 = Y743['R1_PCae_lysoPC_z']
oof = np.full(len(S743), np.nan); fold_sizes = []; sel_count = pd.Series(0, index=Z.columns)
for tr, te in KFold(5, shuffle=True, random_state=20260822).split(np.arange(len(S743))):
    w = select(tr, y1_743, S743); fold_sizes.append(int((w != 0).sum()))
    sel_count += (w != 0).astype(int)
    oof[te] = Z.reindex([S743[i] for i in te]).values @ w
W_full = select(np.arange(len(S743)), y1_743, S743)     # weights from all 743
panel = pd.Series(oof, index=S743)
extra82 = [s for s in S825 if s not in set(S743)]
panel = pd.concat([panel, pd.Series(Z.reindex(extra82).values @ W_full, index=extra82)])
panel = ((panel - panel.reindex(S743).mean()) / panel.reindex(S743).std())
add('panel_fold_min', min(fold_sizes), 'genera', ''); add('panel_fold_max', max(fold_sizes), 'genera', '')
add('n_panel_genera_full', int((W_full != 0).sum()), 'genera', 'weights from all 743')
# 20261030_v65：把每一屬的全樣本統計與各 fold 的 t 一起留下來，X3 之後只讀不算。
FOLD_T = []
for tr, _te in KFold(5, shuffle=True, random_state=20260822).split(np.arange(len(S743))):
    FOLD_T.append(select_stats(tr, y1_743, S743)[1])
FOLD_T = np.vstack(FOLD_T)
B_full, T_full, P_full = select_stats(np.arange(len(S743)), y1_743, S743)
np.savez(f'panel_fold_stats_{ST}.npz', genus=np.array(list(Z.columns), dtype=object),
         fold_t=FOLD_T, t_full=T_full, beta_full=B_full, p_full=P_full,
         folds_selected=sel_count.values)
assert ((np.isfinite(FOLD_T) & (2 * stats.t.sf(np.abs(FOLD_T), 1e6) < 0.01)).sum(0) >= 0).all()

PANG = pd.DataFrame(dict(genus=Z.columns, weight_full=W_full, beta_full=B_full,
                         t_full=T_full, p_full=P_full, folds_selected=sel_count.values))
PANG[PANG.folds_selected > 0].sort_values(['folds_selected', 'genus'], ascending=[False, True]) \
    .to_csv(f'TableS10_panel_genera_{ST}.csv', index=False)
# 全樣本權重的方向必須與全樣本 beta 一致，否則兩套統計已經不是同一個模型
_chk = (W_full != 0)
assert np.array_equal(np.sign(B_full[_chk]), W_full[_chk]), '全樣本權重與 beta 方向不一致'

PRIOR3 = ['NK4A136', 'Christensenellaceae_R-7', 'Enterocloster']
def findg(tok):
    hit = [g for g in Z.columns if tok.lower().replace('_', ' ') in g.lower().replace('_', ' ')]
    if not hit: raise SystemExit(f'pre-specified genus not found: {tok}')
    return hit[0]
p3 = [findg(t) for t in PRIOR3]
prior = (Z[p3[0]] + Z[p3[1]] - Z[p3[2]])
prior = (prior - prior.reindex(S743).mean()) / prior.reindex(S743).std()
add('prior_score_genera', ' + '.join(p3[:2]) + ' - ' + p3[2], '', 'pre-specified 3-genus score')

pan743 = panel.reindex(S743).values; pri743 = prior.reindex(S743).values
pan825 = panel.reindex(S825).values; pri825 = prior.reindex(S825).values

# ======================= clinical variables =======================
tg_mb = rint(np.log(pd.to_numeric(mb.lab_triglyceride, errors='coerce').reindex(S825)))
hb_mb = rint(pd.to_numeric(mb.lab_HbA1c, errors='coerce').reindex(S825).values)
dm_mb = pd.to_numeric(mb.C_DM, errors='coerce').reindex(S825).values
ck_mb = rint(pd.to_numeric(mb.CKM5, errors='coerce').reindex(S825).values)
ck34_mb = (pd.to_numeric(mb.CKM5, errors='coerce').reindex(S825).values >= 3).astype(float)
fb_mb = FB.reindex(S825).values
sub743 = np.array([s in set(S743) for s in S825])
has_med = pd.to_numeric(mb[MED[0]], errors='coerce').reindex(S825).notna().values
Mall = np.nan_to_num(mb[MED].apply(pd.to_numeric, errors='coerce').reindex(S825).values)
Mkey = np.nan_to_num(mb[KEY].apply(pd.to_numeric, errors='coerce').reindex(S825).values)
add('n_with_medication_data', int(has_med.sum()), 'people', 'dx_queried==1 within MB825')
add('n_with_medication_data_743', int((has_med & sub743).sum()), 'people', '')

idx825 = pd.Index(S825)
def s825(mask):  return [S825[i] for i in np.where(mask)[0]]

# ======================= Table 1 (MB825, by CKM stage) =======================
T = mb.reindex(S825).copy(); T['grp'] = 'Stage ' + T.CKM5.astype(int).astype(str)
order = [f'Stage {i}' for i in range(5)]
num_rows = [('age', 'Age, years'), ('BMI', 'Body-mass index, kg/m²'),
            ('waist_cm', 'Waist circumference, cm'), ('SBP', 'Systolic BP, mmHg'),
            ('DBP', 'Diastolic BP, mmHg'), ('lab_triglyceride', 'Triglycerides, mg/dL'),
            ('lab_HDL', 'HDL cholesterol, mg/dL'), ('lab_HbA1c', 'HbA1c, %'),
            ('lab_eGFR', 'eGFR, mL/min/1.73 m²')]
bin_rows = [('sex', 'Female sex'), ('C_DM', 'Diabetes mellitus'),
            ('C_highTG', 'Elevated triglycerides'), ('med_Statin', 'Statin'),
            ('med_Metformin', 'Metformin'), ('med_PPARalpha_Fibrate', 'Fibrate'),
            ('med_SGLT2i', 'SGLT2 inhibitor'), ('med_Insulin', 'Insulin')]
rows = [dict(Characteristic='n', **{g: str((T.grp == g).sum()) for g in order},
             Overall=str(len(T)), P='')]
for c, lab in num_rows:
    v = pd.to_numeric(T[c], errors='coerce'); r = dict(Characteristic=lab); gs = []
    for g in order:
        s_ = v[T.grp == g].dropna(); gs.append(s_)
        r[g] = f'{s_.mean():.1f} ({s_.std():.1f})' if len(s_) else '—'
    r['Overall'] = f'{v.dropna().mean():.1f} ({v.dropna().std():.1f})'
    r['P'] = stats.kruskal(*[x for x in gs if len(x) > 1]).pvalue
    rows.append(r)
med_den = {}
for c, lab in bin_rows:
    v = pd.to_numeric(T[c], errors='coerce'); r = dict(Characteristic=lab); tab = []
    for g in order:
        s_ = v[T.grp == g].dropna()
        r[g] = f'{int(s_.sum())} ({100*s_.mean():.1f})' if len(s_) else '—'
        tab.append([s_.sum(), len(s_) - s_.sum()])
        if c == 'med_Statin': med_den[g] = len(s_)
    r['Overall'] = f'{int(v.dropna().sum())} ({100*v.dropna().mean():.1f})'
    try: r['P'] = stats.chi2_contingency(np.array(tab).T)[1]
    except Exception: r['P'] = ''
    rows.append(r)
pd.DataFrame(rows)[['Characteristic'] + order + ['Overall', 'P']] \
    .to_csv(f'Table1_cohort_{ST}.csv', index=False)
for i in range(5): add(f'n_t1_stage{i}', int((T.CKM5 == i).sum()), 'people', 'MB825')
add('t1_med_denominator_overall', int(pd.to_numeric(T.med_Statin, errors='coerce').notna().sum()),
    'people', '')

# ======================= specificity scan: 139 genera × 8 ratios (MB743) =======================
rows = []
for r in RAT:
    yv = Y743[r]
    for g in Z.columns:
        b, se, p, n = fit(yv, Z[g].reindex(S743).values, S743)
        rows.append(dict(ratio=r, label=RLAB[r], genus=g, beta=b, se=se, p=p, n=n))
SP = pd.DataFrame(rows); SP['q'] = bh(SP.p.values)
SP.to_csv(f'TableS2_specificity_{ST}.csv', index=False)
cnt = SP[SP.p < 0.01].groupby('ratio').size().reindex(RAT).fillna(0).astype(int)
tot = len(SP[SP.p < 0.01])
exp = len(Z.columns) * 0.01
add('n_tests_specificity', len(SP), 'tests', '139 genera x 8 ratios')
add('n_specificity_expected', round(exp, 2), 'genera', 'chance at P<0.01')
for r in RAT: add(f'spec_{r.split("_")[0]}_count', int(cnt[r]), 'genera', 'P<0.01')
add('spec_R1_share', round(100 * cnt['R1_PCae_lysoPC_z'] / max(tot, 1), 1), '%', '')
add('binom_p_R1_concentration',
    float(stats.binomtest(int(cnt['R1_PCae_lysoPC_z']), tot, 1 / 8, 'greater').pvalue),
    '', 'binomial, signal concentrated on R1')
add('n_specificity_total', tot, 'genera', 'P<0.01 across all 8 ratios')
SP[SP.q < 0.05].sort_values('p').to_csv(f'TableS3_specificity_q05_{ST}.csv', index=False)
add('n_specificity_q05', int((SP.q < 0.05).sum()), 'tests', 'BH over 1,112 tests')

# ======================= robustness ladder (MB743) =======================
rob = []
for xn, x825 in [('CV panel', pan825), ('Pre-specified score', pri825)]:
    for yn, yfull in [('PC ae/lysoPC', Y825['R1_PCae_lysoPC_z']), ('log(triglycerides)', tg_mb)]:
        for mlab, ex, wb, msk in [
                ('Age, sex, batch, BMI', [], True, sub743),
                ('+ 8 metabolic drugs', [Mkey], True, sub743 & has_med),
                (f'+ all {len(MED)} medications', [Mall], True, sub743 & has_med),
                ('+ Firmicutes/Bacteroidota ratio', [fb_mb], True, sub743),
                ('+ log(triglycerides)', [tg_mb], True, sub743),
                ('no BMI adjustment', [], False, sub743)]:
            if yn == 'log(triglycerides)' and mlab == '+ log(triglycerides)': continue
            ss = s825(msk); ii = np.where(msk)[0]
            b, se, p, n = fit(yfull[ii], x825[ii], ss, with_bmi=wb,
                              extra=[e[ii] for e in ex] if ex else None)
            rob.append(dict(Exposure=xn, Outcome=yn, Model=mlab, Beta=b, SE=se, P=p, N=n))
ROB = pd.DataFrame(rob); ROB.to_csv(f'TableS6_robustness_{ST}.csv', index=False)
MTAG = [('Age, sex, batch, BMI', 'base'), ('+ 8 metabolic drugs', 'med8'),
        (f'+ all {len(MED)} medications', 'med28'),
        ('+ Firmicutes/Bacteroidota ratio', 'fb'), ('+ log(triglycerides)', 'adjTG'),
        ('no BMI adjustment', 'nobmi')]
for xn, tag in [('CV panel', 'panel'), ('Pre-specified score', 'prior')]:
    for yn, yt in [('PC ae/lysoPC', 'R1'), ('log(triglycerides)', 'TG')]:
        for mlab, mt in MTAG:
            r = ROB[(ROB.Exposure == xn) & (ROB.Outcome == yn) & (ROB.Model == mlab)]
            if not len(r): continue
            add(f'rob_{tag}_{yt}_{mt}_beta', float(r.Beta.iloc[0]), 'SD per SD', 'TableS6_v54')
            add(f'rob_{tag}_{yt}_{mt}_p', float(r.P.iloc[0]), '', 'TableS6_v54')
            add(f'rob_{tag}_{yt}_{mt}_n', int(r.N.iloc[0]), 'people', '')

# ======================= microbiota → clinical, on the full 825 =======================
rows = []
for xn, x in [('CV panel', pan825), ('Pre-specified score', pri825)]:
    for on, yv, bi in [('log(triglycerides)', tg_mb, False), ('HbA1c', hb_mb, False),
                       ('Diabetes', dm_mb, True), ('CKM stage (5-level)', ck_mb, False),
                       ('CKM stage >= 3', ck34_mb, True)]:
        b, se, p, n = fit(yv, x, S825, binary=bi)
        rows.append(dict(exposure=xn, outcome=on, beta=b, se=se, p=p, n=n, set='MB825'))
        b2, se2, p2, n2 = fit(yv[sub743], x[sub743], S743, binary=bi)
        rows.append(dict(exposure=xn, outcome=on, beta=b2, se=se2, p=p2, n=n2, set='MB743'))
MC = pd.DataFrame(rows); MC['q'] = bh(MC.p.values)
MC.to_csv(f'TableS11_microbiota_clinical_{ST}.csv', index=False)
for _, r in MC.iterrows():
    t = ('panel' if r.exposure == 'CV panel' else 'prior') + '_' + \
        r.outcome.replace(' ', '').replace('(', '').replace(')', '').replace('>=', 'ge').replace('-', '')
    for f_, v_ in [('beta', r.beta), ('p', r.p), ('q', r.q), ('n', int(r.n))]:
        add(f'mb2c_{t}_{r["set"]}_{f_}', v_, '', '')

# ======================= species-level decomposition (MB743) =======================
RAW = pd.read_csv(f'{UP}/analysis/04_data_matrices/p180_all.csv', low_memory=False)
RAW['sid'] = RAW.subject_base.astype(str); yr = pd.to_numeric(RAW.test_year, errors='coerce')
PCAE = [c for c in RAW.columns if re.match(r'^PC ae ', c)]
LYSO = [c for c in RAW.columns if re.match(r'^lysoPC ', c)]
def spec_pheno(cols, denom=None):
    out = {}
    for c in cols:
        v = pd.to_numeric(RAW[c], errors='coerce')
        r = v / denom if denom is not None else v
        lr = np.log(r.where(r > 0)); lr = lr - lr.groupby(yr).transform('mean')
        out[c] = lr.groupby(RAW.sid).mean()
    return pd.DataFrame(out)
tot_lyso = RAW[LYSO].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
SP_ratio = spec_pheno(PCAE, tot_lyso)
SP_abs_ae = spec_pheno(PCAE)
SP_abs_ly = spec_pheno(LYSO)
rows = []
for grp, TBL in [('PC ae / total lysoPC', SP_ratio), ('PC ae absolute', SP_abs_ae),
                 ('lysoPC absolute', SP_abs_ly)]:
    for c in TBL.columns:
        yv = rint(TBL[c].reindex(PHE.subject_id).values)
        yv = pd.Series(yv, index=PHE.subject_id).reindex(S743).values
        for sn, x in [('CV panel', pan743), ('Pre-specified score', pri743)]:
            b, se, p, n = fit(yv, x, S743)
            rows.append(dict(group=grp, score=sn, species=c, beta=b, se=se, p=p, n=n))
SPC = pd.DataFrame(rows)
# 20261010_v64 修正：原本用 np.concatenate 把各家族的 q 串起來再「依位置」指派回 SPC。
# SPC 的列是 CV panel 與 Pre-specified score 交錯排列的，各家族的列並不連續，
# 所以每一列拿到的是別人的 q —— 174 列裡有 74 列 q < p。改為依索引指派。
SPC['q'] = np.nan
for _, g in SPC.groupby(['group', 'score'], sort=False):
    SPC.loc[g.index, 'q'] = bh(g.p.values)
assert (SPC.q >= SPC.p - 1e-12).all(), 'BH 指派仍然有錯：出現 q < p'
SPC.to_csv(f'TableS5_species_{ST}.csv', index=False)
d1 = SPC[(SPC.group == 'PC ae / total lysoPC') & (SPC.score == 'CV panel')]
add('n_species_tested', len(d1), 'species', '')
add('n_species_q05', int((d1.q < 0.05).sum()), 'species', '')
add('n_species_same_direction', int((d1.beta > 0).sum()), 'species', '')
w = 1 / d1.se.values ** 2; mu = np.sum(w * d1.beta.values) / np.sum(w)
Q = np.sum(w * (d1.beta.values - mu) ** 2)
add('species_I2', max(0.0, (Q - (len(d1) - 1)) / Q), '', 'random-effects heterogeneity')
pc = SPC[(SPC.group == 'PC ae absolute') & (SPC.score == 'CV panel')].beta.values
ly = SPC[(SPC.group == 'lysoPC absolute') & (SPC.score == 'CV panel')].beta.values
add('n_PCae_species', len(pc), 'species', ''); add('n_lysoPC_species', len(ly), 'species', '')
add('mw_pcae_vs_lysopc_p', float(stats.mannwhitneyu(pc, ly).pvalue), '', 'Mann-Whitney')

# chain length / unsaturation gradient
cn = d1.species.str.extract(r'C(\d+):(\d+)').astype(float)
Xg = np.column_stack([np.ones(len(d1)), cn[0].values, cn[1].values])
bb, ss = hc3(Xg, d1.beta.values)
df = len(d1) - 3
add('carbon_gradient_beta', bb[1], 'per carbon', 'joint model')
add('carbon_gradient_p', 2 * stats.t.sf(abs(bb[1] / ss[1]), df), '', '')
add('doublebond_gradient_beta', bb[2], 'per double bond', '')
add('doublebond_gradient_p', 2 * stats.t.sf(abs(bb[2] / ss[2]), df), '', '')

# ======================= class-level specificity (MB743) =======================
def class_pheno(expr):
    lr = np.log(expr.where(expr > 0)); lr = lr - lr.groupby(yr).transform('mean')
    v = lr.groupby(RAW.sid).mean()
    return pd.Series(rint(v.reindex(PHE.subject_id).values), index=PHE.subject_id).reindex(S743).values
PCAAc = [c for c in RAW.columns if re.match(r'^PC aa ', c)]
t_ae = RAW[PCAE].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
t_aa = RAW[PCAAc].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
t_ly = RAW[LYSO].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
short = [c for c in PCAE if int(re.search(r'C(\d+):', c).group(1)) <= 36]
long_ = [c for c in PCAE if int(re.search(r'C(\d+):', c).group(1)) > 36]
CLS = {'PC ae / PC aa': t_ae / t_aa, 'Total PC ae': t_ae, 'Total PC aa': t_aa,
       'Total lysoPC': t_ly,
       'Short/long PC ae': (RAW[short].apply(pd.to_numeric, errors='coerce').sum(1, min_count=1) /
                            RAW[long_].apply(pd.to_numeric, errors='coerce').sum(1, min_count=1))}
rows = []
for nm, expr in CLS.items():
    yv = class_pheno(expr)
    for sn, x in [('CV panel', pan743), ('Pre-specified score', pri743)]:
        b, se, p, n = fit(yv, x, S743)
        rows.append(dict(phenotype=nm, score=sn, beta=b, se=se, p=p, n=n))
        if sn == 'CV panel':
            k = nm.lower().replace(' / ', '_over_').replace('/', '_').replace(' ', '_')
            add(f'cls_{k}_beta', b, 'SD per SD', ''); add(f'cls_{k}_p', p, '', '')
            add(f'cls_{k}_se', se, '', '')
CLSD = pd.DataFrame(rows); CLSD.to_csv(f'TableS4_class_specificity_{ST}.csv', index=False)

# ======================= metabolite → clinical outcome (METAB) =======================
cl = C.set_index('subject_id')
SM_ = [s for s in PHE.subject_id if s in cl.index]
add('n_metabolome_clinical', len(SM_), 'people', 'metabolomics with a clinical visit')
age_cl = pd.to_numeric(cl.age, errors='coerce'); sex_cl = pd.to_numeric(cl.sex, errors='coerce')
bmi_cl = pd.to_numeric(cl.BMI, errors='coerce')
def fit_cl(y, x, subjects, binary=False, extra=None):
    idx = list(subjects)
    Cv = np.column_stack([np.ones(len(idx)), age_cl.reindex(idx).values,
                          sex_cl.reindex(idx).values, bmi_cl.reindex(idx).values] +
                         ([np.asarray(e, float) for e in extra] if extra else []))
    X = np.column_stack([np.asarray(x, float), Cv]); yv = np.asarray(y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(yv)
    if binary:
        r = sm.GLM(yv[ok], X[ok], family=sm.families.Binomial()).fit(cov_type='HC3')
        return r.params[0], r.bse[0], r.pvalues[0], int(ok.sum())
    b, se = hc3(X[ok], yv[ok])
    return b[0], se[0], 2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - X.shape[1]), int(ok.sum())
tg_cl = rint(np.log(pd.to_numeric(cl.lab_triglyceride, errors='coerce').reindex(SM_)))
hb_cl = rint(pd.to_numeric(cl.lab_HbA1c, errors='coerce').reindex(SM_).values)
dm_cl = pd.to_numeric(cl.C_DM, errors='coerce').reindex(SM_).values
ck_cl = rint(pd.to_numeric(cl.CKM5, errors='coerce').reindex(SM_).values)
ck34_cl = (pd.to_numeric(cl.CKM5, errors='coerce').reindex(SM_).values >= 3).astype(float)
OUTS = [('log(triglycerides)', tg_cl, False, 'none'), ('HbA1c', hb_cl, False, 'none'),
        ('Diabetes', dm_cl, True, 'none'),
        ('CKM stage (5-level)', ck_cl, False, 'contains TG/HDL/glycaemia criteria'),
        ('CKM stage >= 3', ck34_cl, True, 'contains TG/HDL/glycaemia criteria')]
rows = []
for r in RAT:
    x = ph[r + '_int'].reindex(SM_).values
    for on, yv, bi, circ in OUTS:
        b, se, p, n = fit_cl(yv, x, SM_, binary=bi)
        rows.append(dict(ratio=r.split('_')[0], outcome=on, beta=b, se=se, p=p, n=n,
                         circularity=circ, scale='log-odds per SD' if bi else 'SD per SD'))
L = pd.DataFrame(rows); L['q'] = bh(L.p.values)
L.sort_values('p').to_csv(f'TableS8_ratio_outcome_{ST}.csv', index=False)
for _, r in L.iterrows():
    t = f"{r.ratio}_{r.outcome.replace(' ','').replace('(','').replace(')','').replace('>=','ge').replace('-','')}"
    for f_, v_ in [('beta', r.beta), ('se', r.se), ('p', r.p), ('q', r.q), ('n', int(r.n))]:
        add(f'r2o_{t}_{f_}', v_, '', '')

# ======================= mediation (MB743) =======================
rng = np.random.default_rng(20260911)
def mediate(yv825, binary=False, B=2000):
    ii = np.where(sub743)[0]; idx = [S825[i] for i in ii]
    y1 = Y825['R1_PCae_lysoPC_z'][ii]; xv = pan825[ii]; yo = yv825[ii]
    Cv = cov_for(idx)
    ok = np.isfinite(y1) & np.isfinite(xv) & np.isfinite(yo) & np.isfinite(Cv).all(1)
    Cv, y1, xv, yo = Cv[ok], y1[ok], xv[ok], yo[ok]; n = int(ok.sum())
    Xa = np.column_stack([xv, Cv]); Xb = np.column_stack([y1, xv, Cv]); Xt = Xa
    bpath = (lambda Xm, ym: sm.GLM(ym, Xm, family=sm.families.Binomial()).fit().params[0]) \
        if binary else (lambda Xm, ym: np.linalg.lstsq(Xm, ym, rcond=None)[0][0])
    apath = lambda Xm, ym: np.linalg.lstsq(Xm, ym, rcond=None)[0][0]
    ab = apath(Xa, y1) * bpath(Xb, yo)
    tot = (sm.GLM(yo, Xt, family=sm.families.Binomial()).fit().params[0] if binary
           else np.linalg.lstsq(Xt, yo, rcond=None)[0][0])
    bs = []
    for _ in range(B):
        j = rng.integers(0, n, n)
        try: bs.append(apath(Xa[j], y1[j]) * bpath(Xb[j], yo[j]))
        except Exception: pass
    bs = np.array([v for v in bs if np.isfinite(v)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return ab, lo, hi, tot, n
mrows = []
for nm, yv, bi in [('Triglycerides', tg_mb, False), ('HbA1c', hb_mb, False),
                   ('CKM stage', ck_mb, False), ('Diabetes', dm_mb, True)]:
    ab, lo, hi, tot, n = mediate(yv, binary=bi)
    mrows.append(dict(outcome=nm, indirect=ab, lo=lo, hi=hi, total=tot, proportion=ab / tot, n=n))
    k = nm.replace(' ', '')
    for f_, v_ in [('indirect', ab), ('lo', lo), ('hi', hi), ('total', tot),
                   ('proportion', ab / tot), ('n', n)]:
        add(f'med_{k}_{f_}', v_, '', 'bootstrap 2000')
MEDT = pd.DataFrame(mrows); MEDT.to_csv(f'mediation_{ST}.csv', index=False)
add('med_prop_min', round(100 * MEDT.proportion.min()), '%', '')
add('med_prop_max', round(100 * MEDT.proportion.max()), '%', '')

ok = np.isfinite(pan825) & np.isfinite(bmi_mb.reindex(S825).values)
add('corr_panel_bmi', float(np.corrcoef(pan825[ok], bmi_mb.reindex(S825).values[ok])[0, 1]), '', '')
pd.Series(panel, name='panel_oof').rename_axis('subject_id').to_frame() \
    .join(pd.Series(prior, name='score_prior')).to_csv(f'panel_scores_{ST}.csv')

pd.DataFrame(F).to_csv(f'numbers_{ST}.csv', index=False)
print(f'{len(F)} numbers written to numbers_{ST}.csv\n')
print('sets: MB825 =', len(S825), ' MB743 =', len(S743), ' METAB+clinical =', len(SM_))
print('genera tested:', len(Z.columns), ' panel per fold:', fold_sizes)
print('\nspecificity counts at P<0.01:'); print(cnt.rename(index=RLAB).to_string())
print('\nrobustness (CV panel → R1):')
print(ROB[(ROB.Exposure == 'CV panel') & (ROB.Outcome == 'PC ae/lysoPC')].to_string(index=False))
print('\nmicrobiota → clinical:'); print(MC[MC.exposure == 'CV panel'].to_string(index=False))
print('\nclass specificity:'); print(CLSD[CLSD.score == 'CV panel'].to_string(index=False))
print('\nmediation:'); print(MEDT.to_string(index=False))
print('\nR1 → outcomes:'); print(L[L.ratio == 'R1'].to_string(index=False))
