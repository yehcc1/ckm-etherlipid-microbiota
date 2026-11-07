"""Main figures 1-4, rebuilt on the 825/743 analysis sets — 20260911_v54.

Every value is read from the v54 result files; nothing is typed in.
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, re
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from itertools import combinations
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
UP = '/mnt/user-data/uploads/CKM+microbiome'; ST = '20260911_v56'
N = pd.read_csv('numbers_20260911_v56.csv').set_index('key').value
nv = lambda k: float(N[k]); ni = lambda k: int(float(N[k]))
def sci(p, d=1):
    if p >= 0.01: return f'{p:.3g}'
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f'{m:.{d}f} × 10$^{{{e}}}$'
def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
def pan(ax, l, x=-0.18, y=1.05):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)

# =========================== FIGURE 1 ===========================
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) &
       (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
G = pd.read_csv(f'{UP}/analysis/pipeline/geno_panel500.csv.gz', usecols=['subject_id'])
RNA = set(pd.read_csv(f'{UP}/analysis/rnaseq_analysis_ids_20260823_v38.csv').subject_id.astype(str))
LAY = {'Genotype': set(G.subject_id.astype(str)), 'Metabolome': set(PHE.subject_id),
       'Microbiome': set(A.subject_id), 'Transcriptome': RNA}
n_metab, n_mb = len(LAY['Metabolome']), len(LAY['Microbiome'])
n_both = len(LAY['Metabolome'] & LAY['Microbiome'])
n_rna = len(RNA & LAY['Metabolome'] & LAY['Microbiome'])
n_gen = len(LAY['Genotype'] & LAY['Metabolome'] & LAY['Microbiome'])
assert n_both == ni('n_microbiome') and n_mb == ni('n_microbiome_all')

fig = plt.figure(figsize=(7.2, 4.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.28,
                      left=0.015, right=0.985, top=0.92, bottom=0.14)
a = fig.add_subplot(gs[0, 0]); a.axis('off'); a.set_xlim(0, 100); a.set_ylim(0, 100)
a.text(-2, 101, 'A', fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)
a.text(50, 100, 'Metabolic cohort, Chang Gung Memorial Hospital, Keelung',
       ha='center', va='top', fontsize=6.6, color=INK)

def box(x, y, w, h, txt, fc, ec, lw=1.1, fs=6.2):
    a.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0.6,rounding_size=1.4',
                               fc=fc, ec=ec, lw=lw, zorder=2))
    a.text(x + w / 2, y - h / 2, txt, ha='center', va='center', fontsize=fs,
           color=INK, linespacing=1.5, zorder=3)

box(2, 93, 44, 14, f'Targeted plasma\nmetabolomics\nn = {n_metab:,}', '#EAF2FA', BLUE)
box(54, 93, 44, 14, f'Faecal 16S rRNA within\n180 days of a clinical visit\nn = {n_mb:,}',
    '#E7F5EF', GREEN)
for x0 in (24, 76):
    a.annotate('', xy=(50, 74.0), xytext=(x0, 78.0),
               arrowprops=dict(arrowstyle='-|>', lw=1.1, color=GRAY, mutation_scale=9))
box(13, 73, 74, 14, f'BOTH LAYERS   n = {n_both:,}\nprimary analysis set for every model\n'
                    f'with a metabolite outcome or mediator', '#EAF2FA', BLUE, lw=1.7)
a.annotate('', xy=(26, 52.0), xytext=(26, 58.0),
           arrowprops=dict(arrowstyle='-|>', lw=1.1, color=GRAY, mutation_scale=9))
a.annotate('', xy=(74, 52.0), xytext=(74, 58.0),
           arrowprops=dict(arrowstyle='-|>', lw=1.1, color=GRAY, mutation_scale=9))
box(2, 51, 48, 15, f'Genotype available\nn = {n_gen:,}\n(supplementary sensitivity analysis)',
    '#F3F4F6', GRAY, fs=5.8)
box(54, 51, 44, 15, f'Whole-blood RNA sequencing\nn = {n_rna:,}\n(supplementary analyses)',
    '#F3F4F6', GRAY, fs=5.8)
a.text(50, 22, f'Models that need only microbiota and clinical data use all {n_mb:,}\n'
               f'profiled participants; the {n_mb - n_both} without metabolomics enter\n'
               f'no metabolite model.',
       ha='center', va='center', fontsize=5.9, color=MUTED, linespacing=1.7, style='italic')
a.text(50, 4, 'No genome-wide analysis is reported in this manuscript;\n'
              'genotype appears only in the supplement.',
       ha='center', va='center', fontsize=5.9, color=MUTED, linespacing=1.7, style='italic')

b = fig.add_subplot(gs[0, 1])
rows = []
for k in range(1, 5):
    for combo in combinations(LAY, k):
        inc = set.intersection(*[LAY[c] for c in combo])
        exc = set().union(*[LAY[c] for c in LAY if c not in combo]) if k < 4 else set()
        n_ = len(inc - exc)
        if n_: rows.append((combo, n_))
rows.sort(key=lambda r: -r[1])
xs = np.arange(len(rows)); vals = [n_ for _, n_ in rows]
b.bar(xs, vals, 0.68, color=[GREEN if 'Microbiome' in c else BLUE for c, _ in rows])
for x, v in zip(xs, vals):
    b.text(x, v + max(vals) * 0.02, f'{v:,}', ha='center', va='bottom', fontsize=5.9, color=INK)
b.set_ylim(0, max(vals) * 1.16); b.set_xlim(-0.7, len(rows) - 0.3)
b.set_ylabel('Participants in intersection', fontsize=7); b.set_xticks([])
b.spines[['top', 'right', 'bottom']].set_visible(False)
b.tick_params(length=2.5, colors=MUTED); b.yaxis.label.set_color(INK)
b.set_title('Overlap of the four data layers', fontsize=8, color=INK, pad=4)
b.text(-0.30, 1.05, 'B', transform=b.transAxes, fontsize=10, fontweight='bold',
       va='bottom', ha='left', color=INK)
names = list(LAY); dy = max(vals) * 0.105
for j, nm in enumerate(names):
    yy = -dy * (j + 1.1)
    b.text(-0.9, yy, nm, ha='right', va='center', fontsize=6.2, color=MUTED, clip_on=False)
    for x, (combo, _) in zip(xs, rows):
        b.plot(x, yy, 'o', ms=4.6, color=INK if nm in combo else GRID, clip_on=False, zorder=3)
for x, (combo, _) in zip(xs, rows):
    idx = [k for k, c in enumerate(names) if c in combo]
    if len(idx) > 1:
        b.plot([x, x], [-dy * (min(idx) + 1.1), -dy * (max(idx) + 1.1)],
               color=INK, lw=1.1, zorder=2, clip_on=False)
plt.savefig(f'Figure1_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()
print('Figure 1 done')

# =========================== FIGURE 2 ===========================
SP = pd.read_csv(f'TableS2_specificity_20260911_v54.csv')
ROB = pd.read_csv(f'TableS6_robustness_20260911_v54.csv')
SPC = pd.read_csv(f'TableS5_species_20260911_v54.csv')
n743 = ni('n_microbiome')

fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.52, left=0.115, right=0.98,
                      top=0.94, bottom=0.09)
a = fig.add_subplot(gs[0, 0]); pan(a, 'A')
cnt = SP[SP.p < 0.01].groupby('label').size()
lab_order = SP.drop_duplicates('ratio').set_index('ratio').label
cnt = cnt.reindex(lab_order.values).fillna(0).sort_values()
a.barh(np.arange(len(cnt)), cnt.values, 0.62,
       color=[GREEN if l.startswith('R1') else GRAY for l in cnt.index])
a.axvline(nv('n_specificity_expected'), color=ORANGE, ls='--', lw=1.0)
a.text(nv('n_specificity_expected') + 0.6, 2.6, 'expected\nby chance', fontsize=6,
       color=ORANGE, va='center')
a.set_yticks(np.arange(len(cnt))); a.set_yticklabels(cnt.index, fontsize=6.2)
a.set_xlabel(f'Genera with $P$ < 0.01  (of {ni("n_genera_tested")} tested)')
a.set_title('Microbial signal concentrates on ether lipids', fontsize=8, color=INK, pad=4)
a.text(0.97, 0.06, f'binomial $P$ = {sci(nv("binom_p_R1_concentration"))}',
       transform=a.transAxes, ha='right', fontsize=6.6, color=INK)
clean(a)

b = fig.add_subplot(gs[0, 1]); pan(b, 'B')
sel = ROB[(ROB.Exposure == 'CV panel') & (ROB.Outcome == 'PC ae/lysoPC')]
order = ['Age, sex, batch, BMI', '+ 8 metabolic drugs',
         [m for m in sel.Model if m.startswith('+ all')][0],
         '+ Firmicutes/Bacteroidota ratio', '+ log(triglycerides)']
short = {order[0]: 'Base model + BMI', order[1]: '+ 8 metabolic drugs',
         order[2]: '+ all 28 medications', order[3]: '+ Firmicutes/Bacteroidota',
         order[4]: '+ log(triglycerides)'}
sel = sel.set_index('Model').loc[order]
y = np.arange(len(sel))[::-1]
b.errorbar(sel.Beta, y, xerr=1.96 * sel.SE, fmt='o', ms=4.5, lw=1.1, color=GREEN,
           ecolor=GREEN, capsize=2)
b.axvline(0, color=GRAY, lw=0.8, ls='--')
for yy, (_, r) in zip(y, sel.iterrows()):
    txt = f'$P$ = {sci(r.P)}' + (f'  (n = {int(r.N)})' if int(r.N) != n743 else '')
    b.text(r.Beta + 1.96 * r.SE + 0.012, yy, txt, va='center', fontsize=5.9, color=INK)
b.set_yticks(y); b.set_yticklabels([short[m] for m in sel.index], fontsize=6.2)
b.set_xlim(-0.02, 0.34); b.set_xlabel('Effect on ether-lipid ratio (SD per SD)')
b.set_title(f'Robust to medication and composition (n = {n743})', fontsize=8, color=INK, pad=4)
clean(b)

c = fig.add_subplot(gs[1, 0]); pan(c, 'C')
d1 = SPC[(SPC.group == 'PC ae / total lysoPC') & (SPC.score == 'CV panel')].sort_values('beta')
yy = np.arange(len(d1))
c.errorbar(d1.beta, yy, xerr=1.96 * d1.se, fmt='none', lw=0.8, ecolor=GRID, zorder=1)
c.scatter(d1.beta, yy, s=11, zorder=2,
          c=[GREEN if q < 0.05 else 'white' for q in d1.q],
          edgecolors=GREEN, linewidths=0.8)
c.axvline(0, color=GRAY, lw=0.8, ls='--')
c.set_yticks([]); c.set_xlabel('Effect on species / total lysoPC (SD per SD)')
c.set_ylabel(f'{len(d1)} ether phosphatidylcholine species')
c.set_title(f'{ni("n_species_q05")} of {ni("n_species_tested")} species individually '
            f'significant', fontsize=8, color=INK, pad=4)
c.text(0.03, 0.94, f'all {ni("n_species_same_direction")} in the same direction\n'
                   f'$I^2$ = {100*nv("species_I2"):.0f}%   n = {n743}',
       transform=c.transAxes, fontsize=6.2, color=INK, va='top')
clean(c)

d = fig.add_subplot(gs[1, 1]); pan(d, 'D')
pc = SPC[(SPC.group == 'PC ae absolute') & (SPC.score == 'CV panel')].beta.values
ly = SPC[(SPC.group == 'lysoPC absolute') & (SPC.score == 'CV panel')].beta.values
parts = d.violinplot([pc, ly], positions=[0, 1], widths=0.7, showextrema=False)
for pcol, col in zip(parts['bodies'], [GREEN, PURPLE]):
    pcol.set_facecolor(col); pcol.set_alpha(0.28); pcol.set_edgecolor(col); pcol.set_lw(0.9)
for i, (v, col) in enumerate([(pc, GREEN), (ly, PURPLE)]):
    d.scatter(np.full(len(v), i) + np.random.default_rng(1).normal(0, 0.045, len(v)), v,
              s=8, color=col, alpha=0.75, zorder=3)
    d.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1.4, zorder=4)
d.axhline(0, color=GRAY, lw=0.8, ls='--')
d.set_xticks([0, 1])
d.set_xticklabels([f'PC ae\n({len(pc)} species)', f'lysoPC\n({len(ly)} species)'], fontsize=6.6)
d.set_ylabel('Effect on absolute concentration (SD per SD)')
d.set_title('Ether species rise, lysoPC species do not', fontsize=8, color=INK, pad=4)
d.text(0.97, 0.05, f'Mann–Whitney $P$ = {sci(nv("mw_pcae_vs_lysopc_p"))}\nn = {n743}',
       transform=d.transAxes, ha='right', fontsize=6.4, color=INK)
clean(d)
plt.savefig(f'Figure2_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()
print('Figure 2 done')

# =========================== FIGURE 3 ===========================
CLS = pd.read_csv(f'TableS4_class_specificity_20260911_v54.csv')
fig = plt.figure(figsize=(7.2, 7.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.02, 1.0], hspace=0.26, wspace=0.42,
                      left=0.10, right=0.98, top=0.94, bottom=0.09)
a = fig.add_subplot(gs[0, :]); a.axis('off'); a.set_xlim(0, 100); a.set_ylim(7, 116)
a.text(-1, 116, 'A', fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)
MINUS = lambda t: t
bet = lambda v: f'{v:+.3f}'.replace('-', '\u2212')
def nd(x, y, w, h, txt, fc, ec, fs=6.2):
    a.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0.5,rounding_size=1.2',
                               fc=fc, ec=ec, lw=1.0, zorder=2))
    a.text(x + w / 2, y - h / 2, MINUS(txt), ha='center', va='center', fontsize=fs,
           color=INK, linespacing=1.4, zorder=3)
def arr(x1, y1, x2, y2, col=GRAY, lw=1.0):
    a.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='-|>', lw=lw, color=col, mutation_scale=9))
nd(2, 114, 40, 15, 'GUT BACTERIA  (predicted metagenome)\npalmitate biosynthesis II, type II fatty acid synthase\n→ fatty alcohol supply for the sn-1 position', '#E7F5EF', GREEN, fs=6.0)
a.annotate('', xy=(14, 75), xytext=(14, 98),
           arrowprops=dict(arrowstyle='-|>', lw=1.6, color=GREEN, mutation_scale=11))
a.text(16.5, 88, f'β = {bet(nv("pwy5971_ether-lipid_ratio_beta"))}\n$P$ = {sci(nv("pwy5971_ether-lipid_ratio_p"))}\non the ether-lipid ratio',
       fontsize=5.8, color='#14634A', ha='left', va='center', linespacing=1.4)
nd(36, 96, 28, 12, 'Dihydroxyacetone phosphate', '#F3F4F6', GRAY)
arr(44, 84, 24, 74); arr(56, 84, 76, 74)
a.text(36, 80, 'peroxisome', fontsize=5.8, color=GREEN, style='italic', ha='center')
a.text(72, 79, 'endoplasmic reticulum', fontsize=5.8, color=PURPLE, style='italic', ha='center')
nd(4, 74, 40, 13, 'GNPAT, AGPS  (+ FAR1 fatty alcohol)\n→ sn-1 ETHER bond', '#E7F5EF', GREEN)
nd(56, 74, 40, 13, 'GPAT, AGPAT\n→ sn-1 ester bond', '#F0EEF6', PURPLE)
arr(24, 61, 24, 50); arr(76, 61, 76, 50)
a.text(50, 55.5, 'shared downstream steps:\nsn-2 acylation (LPCAT), choline head group',
       ha='center', va='center', fontsize=5.8, color=MUTED, style='italic', linespacing=1.4)
cv = CLS[CLS.score == 'CV panel'].set_index('phenotype')
nd(4, 50, 40, 13, f'Ether phosphatidylcholine (PC ae)\nβ = {bet(cv.loc["Total PC ae","beta"])}, '
                  f'$P$ = {sci(cv.loc["Total PC ae","p"])}', '#E7F5EF', GREEN)
nd(56, 50, 40, 13, f'Diacyl phosphatidylcholine (PC aa)\nβ = {bet(cv.loc["Total PC aa","beta"])}, '
                   f'$P$ = {sci(cv.loc["Total PC aa","p"])}', '#F0EEF6', PURPLE)
arr(24, 37, 42, 27); arr(76, 37, 58, 27)
nd(26, 27, 48, 13, f'PC ae / PC aa ratio\nβ = {bet(cv.loc["PC ae / PC aa","beta"])}, '
                   f'$P$ = {sci(cv.loc["PC ae / PC aa","p"])}', '#EAF2FA', BLUE, fs=6.8)
a.text(50, 9, f'The gut microbiota panel and the predicted bacterial fatty-acid pathway both move the ether branch\nand not the ester branch (PC aa $P$ = {sci(nv("pwy5971_total_pc_aa_p"))} for the pathway); n = {n743}.',
       ha='center', va='center', fontsize=6.2, color=INK, linespacing=1.5)

b = fig.add_subplot(gs[1, 0]); pan(b, 'B')
ordc = ['PC ae / PC aa', 'Total PC ae', 'Total PC aa', 'Total lysoPC', 'Short/long PC ae']
s = cv.loc[ordc]; y = np.arange(len(s))[::-1]
b.errorbar(s.beta, y, xerr=1.96 * s.se, fmt='o', ms=4.5, lw=1.1,
           color=BLUE, ecolor=BLUE, capsize=2)
b.axvline(0, color=GRAY, lw=0.8, ls='--')
for yy, (_, r) in zip(y, s.iterrows()):
    b.text(r.beta + 1.96 * r.se + 0.012, yy, f'$P$ = {sci(r.p)}', va='center',
           fontsize=5.9, color=INK)
b.set_yticks(y); b.set_yticklabels(ordc, fontsize=6.4)
b.set_xlabel('Effect of microbiota panel (SD per SD)'); b.set_xlim(-0.16, 0.34)
b.set_title(f'Class-level specificity (n = {n743})', fontsize=8, color=INK, pad=4)
clean(b)

c = fig.add_subplot(gs[1, 1]); pan(c, 'C')
d1 = SPC[(SPC.group == 'PC ae / total lysoPC') & (SPC.score == 'CV panel')].copy()
cn = d1.species.str.extract(r'C(\d+):(\d+)').astype(float)
d1['carbon'] = cn[0].values; d1['db'] = cn[1].values
c.scatter(d1.carbon + np.random.default_rng(2).normal(0, 0.12, len(d1)), d1.beta, s=16,
          c=[GREEN if q < 0.05 else 'white' for q in d1.q], edgecolors=GREEN, linewidths=0.8)
z = np.polyfit(d1.carbon, d1.beta, 1); xx = np.linspace(d1.carbon.min(), d1.carbon.max(), 20)
c.plot(xx, np.polyval(z, xx), color=ORANGE, lw=1.2)
c.axhline(0, color=GRAY, lw=0.8, ls='--')
c.set_xlabel('PC ae carbon number'); c.set_ylabel('Effect (SD per SD)')
c.set_title('Graded by chain length, not unsaturation', fontsize=8, color=INK, pad=4)
c.text(0.03, 0.05, f'carbon number $P$ = {sci(nv("carbon_gradient_p"))}\n'
                   f'double bonds $P$ = {nv("doublebond_gradient_p"):.2f}   n = {n743}',
       transform=c.transAxes, fontsize=6.2, color=INK, va='bottom')
clean(c)
plt.savefig(f'Figure3_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()
print('Figure 3 done')

# =========================== FIGURE 4 ===========================
S8 = pd.read_csv(f'TableS8_ratio_outcome_20260911_v54.csv')
MEDT = pd.read_csv(f'mediation_20260911_v54.csv')
MC = pd.read_csv(f'TableS11_microbiota_clinical_20260911_v54.csv')
fig = plt.figure(figsize=(7.2, 2.9))
gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.11, right=0.985, top=0.84, bottom=0.22)

a = fig.add_subplot(gs[0, 0]); pan(a, 'A', x=-0.36)
r1 = S8[(S8.ratio == 'R1') & (S8.scale == 'SD per SD')].set_index('outcome')
ordr = ['log(triglycerides)', 'CKM stage (5-level)', 'HbA1c']
s = r1.loc[ordr]; y = np.arange(len(s))[::-1]
a.errorbar(s.beta, y, xerr=1.96 * s.se, fmt='o', ms=4.5, lw=1.1, color=BLUE, ecolor=BLUE, capsize=2)
a.axvline(0, color=GRAY, lw=0.8, ls='--')
for yy, (_, r) in zip(y, s.iterrows()):
    a.text(r.beta, yy + 0.20, f'$q$ = {sci(r.q)}', va='bottom', ha='center',
           fontsize=5.8, color=INK)
a.set_yticks(y); a.set_yticklabels(['Triglycerides', 'CKM stage', 'HbA1c'], fontsize=6.6)
a.set_xlim(-0.46, 0.10); a.set_ylim(-0.5, 2.75); a.set_xlabel('Effect of ether-lipid ratio (SD per SD)')
dmr = S8[(S8.ratio == 'R1') & (S8.outcome == 'Diabetes')].iloc[0]
a.set_title(f'Ether-lipid ratio tracks phenotype\n(n = {int(s.n.min()):,} to {int(s.n.max()):,})',
            fontsize=7.6, color=INK, pad=4)
a.text(0.5, -0.40, f'Diabetes: odds ratio {np.exp(dmr.beta):.2f} per SD, $q$ = {sci(dmr.q)}',
       transform=a.transAxes, ha='center', fontsize=6.0, color=INK)
clean(a)

b = fig.add_subplot(gs[0, 1]); pan(b, 'B', x=-0.36)
y = np.arange(len(MEDT))[::-1]
b.errorbar(MEDT.indirect, y, xerr=[MEDT.indirect - MEDT.lo, MEDT.hi - MEDT.indirect],
           fmt='o', ms=4.5, lw=1.1, color=GREEN, ecolor=GREEN, capsize=2)
b.axvline(0, color=GRAY, lw=0.8, ls='--')
for yy, (_, r) in zip(y, MEDT.iterrows()):
    b.text(r.indirect, yy + 0.20, f'{100*r.proportion:.0f}% mediated', va='bottom',
           ha='center', fontsize=6.0, color=INK)
b.set_yticks(y); b.set_yticklabels(MEDT.outcome, fontsize=6.6)
b.set_xlabel('Indirect effect (95% CI)'); b.set_xlim(-0.125, 0.030); b.set_ylim(-0.5, 3.75)
b.set_title(f'Mediated by the ether-lipid ratio\n(2,000 resamples, n = {n743})',
            fontsize=7.6, color=INK, pad=4)
clean(b)

c = fig.add_subplot(gs[0, 2]); pan(c, 'C', x=-0.36)
m = MC[(MC.exposure == 'CV panel') & (MC['set'] == 'MB825')].set_index('outcome')
ordm = ['log(triglycerides)', 'HbA1c', 'CKM stage (5-level)']
s = m.loc[ordm]; y = np.arange(len(s))[::-1]
c.errorbar(s.beta, y, xerr=1.96 * s.se, fmt='o', ms=4.5, lw=1.1, color=ORANGE,
           ecolor=ORANGE, capsize=2)
c.axvline(0, color=GRAY, lw=0.8, ls='--')
for yy, (_, r) in zip(y, s.iterrows()):
    c.text(r.beta, yy + 0.20, f'$q$ = {sci(r.q)}', va='bottom', ha='center',
           fontsize=5.8, color=INK)
c.set_yticks(y); c.set_yticklabels(['Triglycerides', 'HbA1c', 'CKM stage'], fontsize=6.6)
c.set_xlim(-0.32, 0.08); c.set_ylim(-0.5, 2.75); c.set_xlabel('Effect of microbiota panel (SD per SD)')
dm = MC[(MC.exposure == 'CV panel') & (MC['set'] == 'MB825') &
        (MC.outcome == 'Diabetes')].iloc[0]
c.set_title(f'Microbiota tracks phenotype in all\n{ni("n_microbiome_all")} profiled participants',
            fontsize=7.6, color=INK, pad=4)
c.text(0.5, -0.40, f'Diabetes: odds ratio {np.exp(dm.beta):.2f} per SD, $q$ = {sci(dm.q)}',
       transform=c.transAxes, ha='center', fontsize=6.0, color=INK)
clean(c)
plt.savefig(f'Figure4_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()
print('Figure 4 done')
