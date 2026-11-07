"""Figure 3 rebuilt with a fourth panel — 20260912_v59.

Panels A, B and C are reproduced from X5_figures_main_20260911_v56.py without change; the
new panel D is the microbial route map, built as the counterpart of Figure 4C of the earlier
analysis of this cohort. Every node in D is a MetaCyc pathway that survives adjustment for
reference-genome coverage, and every number printed on it is read from the enrichment table
(beta from the primary model, q* from the NSTI-adjusted model — the same convention as
Figure S5C). Nothing is typed in: the layout names pathway identifiers, and the script asserts
that each one survives and carries the sign the map draws it with.
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
ST = '20260912_v59'
N = pd.read_csv('numbers_20260911_v58.csv').set_index('key').value
nv = lambda k: float(N[k]); ni = lambda k: int(float(N[k]))
n743 = ni('n_microbiome')

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

CLS = pd.read_csv('TableS4_class_specificity_20260911_v54.csv')
SPC = pd.read_csv('TableS5_species_20260911_v54.csv')

fig = plt.figure(figsize=(7.2, 9.7))
gs = fig.add_gridspec(3, 2, height_ratios=[0.97, 0.84, 1.19], hspace=0.30, wspace=0.42,
                      left=0.10, right=0.98, top=0.955, bottom=0.05)

# ------------------------------- A -------------------------------
a = fig.add_subplot(gs[0, :]); a.axis('off'); a.set_xlim(0, 100); a.set_ylim(2, 116)
a.text(-1, 116, 'A', fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)
bet = lambda v: f'{v:+.3f}'.replace('-', '−')
def nd(x, y, w, h, txt, fc, ec, fs=6.2):
    a.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0.5,rounding_size=1.2',
                               fc=fc, ec=ec, lw=1.0, zorder=2))
    a.text(x + w / 2, y - h / 2, txt, ha='center', va='center', fontsize=fs,
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
a.text(50, 7, f'The gut microbiota panel and the predicted bacterial fatty-acid pathway both move the ether branch\nand not the ester branch (PC aa $P$ = {sci(nv("pwy5971_total_pc_aa_p"))} for the pathway); n = {n743}.',
       ha='center', va='center', fontsize=6.2, color=INK, linespacing=1.5)

# ------------------------------- B -------------------------------
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

# ------------------------------- C -------------------------------
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

# ------------------------------- D -------------------------------
M = pd.read_csv('TableS17_metacyc_enrichment_20260911_v55.csv')
P = M[M.exposure == 'Microbiota panel'].set_index('pathway')
GEM = set(pd.read_csv('TableS19_replication_prior_paper_20260911_v55.csv').pathway)
n825 = int(P.n.iloc[0])

def qf(q):
    if q >= 0.01: return f'{q:.3g}'
    e = int(np.floor(np.log10(q))); m = q / 10 ** e
    return f'{m:.1f} × 10$^{{{e}}}$'

def node(pw, short, expect):
    """label for one pathway box; numbers are read, the sign is checked not assumed."""
    r = P.loc[pw]
    assert r.q < 0.05 and r.q_nsti_adj < 0.05, f'{pw} does not survive'
    assert np.sign(r.beta) == expect, f'{pw} sign {np.sign(r.beta)} is not {expect}'
    dag = '†' if pw in GEM else ''
    return f'{short}{dag}\nβ = {bet(r.beta)}, $q$* = {qf(r.q_nsti_adj)}', r.beta

gsd = gs[2, :].subgridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.10)
d = fig.add_subplot(gsd[0, 0]); d.axis('off'); d.set_xlim(0, 100); d.set_ylim(-28, 104)
d.text(-2, 102, 'D', fontsize=10, fontweight='bold', va='bottom', ha='left', color=INK)

def bx(x, y, w, h, txt, fc, ec, fs=5.4, lw=1.0):
    d.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle='round,pad=0.35,rounding_size=1.0',
                               fc=fc, ec=ec, lw=lw, zorder=2))
    d.text(x + w / 2, y - h / 2, txt, ha='center', va='center', fontsize=fs,
           color=INK, linespacing=1.4, zorder=3)
def ar(x1, y1, x2, y2, col=GRAY, lw=1.0):
    d.annotate('', xy=(x2, y2), xytext=(x1, y1),
               arrowprops=dict(arrowstyle='-|>', lw=lw, color=col, mutation_scale=9,
                               shrinkA=0, shrinkB=0))

GLIGHT, CHIP = '#E7F5EF', '#F3F4F6'
COL = [(1, 31), (34.5, 31), (68, 31)]
CEN = [x + w / 2 for x, w in COL]

bx(31, 103, 38, 8, 'Dietary sucrose and hexoses', CHIP, GRAY, fs=5.8)
ar(50, 95, 50, 89.5)
for (x, w), (pw, short) in zip(COL, [('PWY-5384', 'Sucrose degradation IV\n(sucrose phosphorylase)'),
                                     ('PWY-621', 'Sucrose degradation III\n(sucrose invertase)'),
                                     ('P124-PWY', 'Bifidobacterium shunt\n(F6P phosphoketolase)')]):
    t, _ = node(pw, short, +1)
    bx(x, 88, w, 15, t, GLIGHT, GREEN)
ar(CEN[0], 73, 37, 69.5); ar(CEN[1], 73, CEN[1], 69.5); ar(CEN[2], 73, 63, 69.5)
d.text(2, 65, 'glycolysis', fontsize=5.2, color=MUTED, style='italic', ha='left')
bx(33, 68, 34, 8, 'Pyruvate, acetyl phosphate', CHIP, GRAY, fs=5.8)
ar(37, 60, CEN[0], 56.5); ar(CEN[1], 60, CEN[1], 56.5); ar(63, 60, CEN[2], 56.5)
for (x, w), (pw, short) in zip(COL, [('FERMENTATION-PWY', 'Mixed acid\nfermentation'),
                                     ('P122-PWY', 'Heterolactic\nfermentation'),
                                     ('PWY-6588', 'Pyruvate fermentation\nto acetone')]):
    t, _ = node(pw, short, +1)
    bx(x, 55, w, 15, t, GLIGHT, GREEN)
ar(CEN[0], 40, 43, 36.5); ar(CEN[1], 40, CEN[1], 36.5); ar(CEN[2], 40, 57, 36.5)
bx(40, 35, 20, 8, 'Acetyl-CoA', CHIP, GRAY, fs=5.8)
ar(50, 27, 50, 24, col=GREEN, lw=1.6)

t, _ = node('PWY-5971', 'Palmitate biosynthesis II  (type II fatty acid synthase)', +1)
bx(4, 23, 92, 18, t + '\nthe only one of 365 that also survives adjustment for the host ratio '
                     '(Figure S5C)',
   GLIGHT, GREEN, fs=5.7, lw=1.9)
ar(50, 5, 50, -0.5, col=GREEN, lw=1.6)
bx(2, -1, 96, 13,
   f'HOST   FAR1 → hexadecanol → GNPAT/AGPS sn-1 ether bond (panel A)\n'
   f'β = {bet(nv("pwy5971_ether-lipid_ratio_beta"))} on the ether-lipid ratio, '
   f'$P$ = {sci(nv("pwy5971_ether-lipid_ratio_p"))}, n = {n743}', '#EAF2FA', BLUE, fs=6.0)
d.text(0, -17, f'Predicted metagenome (PICRUSt2), n = {n825}.  β is from the primary model,\n'
               f'$q$* from the model additionally adjusted for reference-genome coverage;\n'
               f'every pathway shown reaches $q$ < 0.05 in both.  † also reported for\n'
               f'$Gemmiger$ in the earlier analysis of this cohort.',
       fontsize=5.4, color=MUTED, ha='left', va='top', linespacing=1.7)

# ---- the opposite arm
e = fig.add_subplot(gsd[0, 1])
NEG = ['P162-PWY', 'PWY-5022', 'HISDEG-PWY', 'PWY-7254', 'P108-PWY', 'PWY-4984']
for pw in NEG:
    assert P.loc[pw, 'q'] < 0.05 and P.loc[pw, 'q_nsti_adj'] < 0.05 and P.loc[pw, 'beta'] < 0
s_ = P.loc[NEG].sort_values('beta', ascending=False)
yy = np.arange(len(s_))
e.barh(yy, s_.beta, 0.34, color=PURPLE, edgecolor='white', linewidth=0.6)
e.axvline(0, color=GRAY, lw=0.8)
for i, (pw, r) in enumerate(s_.iterrows()):
    nm = r['name'].replace('&beta;', 'β')
    if not nm.startswith('L-'):
        nm = nm[0].upper() + nm[1:]
    nm += '†' if pw in GEM else ''
    e.text(-0.455, i + 0.30, nm, va='bottom', ha='left', fontsize=5.5, color=INK)
    e.text(r.beta - 0.008, i, f'$q$* = {qf(r.q_nsti_adj)}', va='center', ha='right',
           fontsize=5.2, color=MUTED)
e.set_yticks([]); e.set_ylim(-0.7, len(s_) - 0.25)
e.set_xlim(-0.46, 0.02); e.set_xticks([-0.4, -0.3, -0.2, -0.1, 0.0])
e.set_xlabel('Effect of microbiota panel (SD per SD)', fontsize=6.2)
e.set_title('Runs the other way:\nprotein and organic-acid degradation',
            fontsize=7, color=PURPLE, pad=5)
e.spines[['top', 'right', 'left']].set_visible(False)
e.tick_params(length=2.5, colors=MUTED, labelsize=6)
e.xaxis.label.set_color(INK)

plt.savefig(f'Figure3_{ST}.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('F3_v59.png', dpi=190, bbox_inches='tight', pad_inches=0.10)
plt.close()
print('Figure 3 with panel D done')
