# -*- coding: utf-8 -*-
"""Figure S2 改成兩種暴露並列（事前分數為主）    20261030_v65"""
"""Supplementary Figures S1 and S2 rebuilt on the v54 samples — 20260911.

S1  properties of the eight ratio phenotypes, now across all 3,688 profiled participants
S2  each principal result re-estimated in the progressively smaller, fully overlapping
    subsets — which are different subsets in v54, so this figure had to be rebuilt

S3 to S6 are carried forward from v45 unchanged: their inputs (the pre-specified functional
guild lists, the PICRUSt2 output and the RNA-sequencing matrix) are not in this working set,
so they could not be re-estimated here. That is recorded in the author notes and guarded by
the verifier, and the three numbers the text quotes from Figure S5 are labelled in Methods
as coming from the earlier sample.
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from xnames import labels as taxa_labels
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
UP = '/mnt/user-data/uploads/CKM+microbiome'
HUB = '/mnt/user-data/uploads/microbiome_hub_v1/data'
ST = '20261030_v65'
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
def clean(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
def pan(ax, l, x=-0.20, y=1.04):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)

PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
RAT = ['R1_PCae_lysoPC_z', 'R2_PCae362_361_z', 'R3_SM_over_PC_z', 'R4_SM240_160_z',
       'R5_lysoPC204_PCaa384_z', 'R6_PCaa384_383_z', 'R7_C16_C0_z', 'R8_C2_C0_z']
LAB = ['R1 PC ae/lysoPC', 'R2 PC ae 36:2/36:1', 'R3 SM/PC', 'R4 SM 24:0/16:0',
       'R5 lysoPC 20:4/PC aa 38:4', 'R6 PC aa 38:4/38:3', 'R7 C16/C0', 'R8 C2/C0']
Y = pd.DataFrame({r: rint(PHE[r].values) for r in RAT}, index=PHE.subject_id)

# ============================== FIGURE S2 ==============================
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
mb = A.set_index('subject_id'); S825 = list(A.subject_id)
S743 = [s for s in S825 if s in set(PHE.subject_id)]
GENO = set(pd.read_csv(f'{UP}/analysis/pheno_final_20260822_v25.csv').subject_id.astype(str))
S585 = [s for s in S743 if s in GENO]
SC = pd.read_csv(f'out/panel_scores_{ST}.csv'); SC['subject_id'] = SC.subject_id.astype(str)
SC = SC.set_index('subject_id')
age = pd.to_numeric(mb.age, errors='coerce'); sex = pd.to_numeric(mb.sex, errors='coerce')
bmi = pd.to_numeric(mb.BMI, errors='coerce')
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float)
y1 = Y['R1_PCae_lysoPC_z']
tg = pd.Series(rint(np.log(pd.to_numeric(mb.lab_triglyceride, errors='coerce').reindex(S825))),
               index=S825)

def est(y, x, subj):
    Cv = np.column_stack([np.ones(len(subj)), age.reindex(subj).values,
                          sex.reindex(subj).values, bmi.reindex(subj).values,
                          BAT.reindex(subj).values])
    X = np.column_stack([np.asarray(x, float), Cv]); yv = np.asarray(y, float)
    ok = np.isfinite(X).all(1) & np.isfinite(yv); X, yv = X[ok], yv[ok]
    keep = np.r_[True, X[:, 1:].std(0) > 1e-10]
    keep[1] = True                      # the intercept has zero variance by construction
    X = X[:, keep]
    b, se = hc3(X, yv)
    return b[0], se[0], 2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - X.shape[1]), int(ok.sum())

SUB = [('All 16S profiled', S825), ('+ metabolomics', S743),
       ('+ genetic principal components', S585)]
fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.8))
fig.subplots_adjust(wspace=0.95, left=0.14, right=0.86, top=0.84, bottom=0.26)
rows = []
for ax, (lab, ycol) in zip(axes, [('Microbiota score → ether-lipid ratio', 'R1'),
                                  ('Microbiota score → log(triglycerides)', 'TG')]):
    pan(ax, 'AB'[0 if ycol == 'R1' else 1], x=-0.34)
    yy = np.arange(len(SUB))[::-1]
    for ex, scol, col, dy, mk in [('Pre-specified score', 'score_prior', GREEN, +0.17, 'o'),
                                  ('CV panel', 'panel_oof', GRAY, -0.17, 's')]:
        for k, (nm, subj) in zip(yy, SUB):
            if ycol == 'R1' and nm == 'All 16S profiled':
                if ex == 'CV panel':
                    ax.text(0.5, k, 'not estimable — no metabolomics', transform=ax.get_yaxis_transform(),
                            va='center', ha='center', fontsize=5.6, color=MUTED, style='italic')
                continue
            y = (y1.reindex(subj) if ycol == 'R1' else tg.reindex(subj)).values
            b, se, p, n = est(y, SC[scol].reindex(subj).values, subj)
            rows.append(dict(outcome=ycol, exposure=ex, subset=nm, beta=b, se=se, p=p, n=n))
            ax.errorbar(b, k + dy, xerr=1.96 * se, fmt=mk, ms=4.0, lw=1.0, color=col, ecolor=col, capsize=1.8, label=ex if k == yy[-1] else None)
            ax.text(1.02, k + dy, f'$P$ = {p:.2g}  (n = {n})', va='center', ha='left', transform=ax.get_yaxis_transform(),
                    fontsize=5.4, color=INK if ex == 'Pre-specified score' else MUTED)
    ax.axvline(0, color=GRAY, lw=0.8, ls='--')
    ax.set_yticks(yy); ax.set_yticklabels([v[0] for v in SUB], fontsize=6.4)
    ax.set_ylim(-0.7, len(SUB) - 0.3)
    ax.legend(frameon=False, fontsize=5.6, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, labelcolor=MUTED, handletextpad=0.3)
    ax.set_xlabel('Effect (SD per SD)')
    ax.set_title(lab, fontsize=7.6, color=INK, pad=4)
    clean(ax)
pd.DataFrame(rows).to_csv(f'out/figS2_subsets_{ST}.csv', index=False)
fig.suptitle('Each principal result re-estimated in progressively smaller, fully overlapping subsets',
             fontsize=8.0, color=INK, y=1.02)
plt.savefig(f'FigureS2_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.savefig('chk/S2_v65.png', dpi=170, bbox_inches='tight'); plt.close()
print('Figure S2 done')
