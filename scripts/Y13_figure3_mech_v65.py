"""Figure 3（機制圖，三 panel A–C）改以事前三菌屬分數為主 — 20261031_v65。

20261031 第二次修訂：原 panel D（方塊路徑圖＋反向臂長條）撤除，改由 Y15 的代謝物層級路徑圖（Figure 4，仿 Wu 2025 Figure 4c）取代。

原 20260912_v59 四 panel 版本；數字改讀 v65 結果檔與 TableSmc（X28：事前分數與三個事前菌屬各自的 MetaCyc 富集）。
D 的每個節點標示是哪一個事前菌屬扛的（NK4A136／Christensenellaceae R-7），右欄反向臂標示 Enterocloster。

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
ST = '20261031_v65'
N = pd.concat([pd.read_csv('numbers_v65_all_20261030_v65.csv'), pd.read_csv('numbers_extra_20261030_v65.csv')]).drop_duplicates('key', keep='last').set_index('key').value
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

CLS = pd.read_csv('out/TableS4_class_specificity_20261030_v65.csv')
SPC = pd.read_csv('out/TableS5_species_20261030_v65.csv')

fig = plt.figure(figsize=(7.4, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[0.97, 0.84], hspace=0.30, wspace=0.42,
                      left=0.10, right=0.98, top=0.95, bottom=0.08)

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
cv = CLS[CLS.score == 'Pre-specified score'].set_index('phenotype')
nd(4, 50, 40, 13, f'Ether phosphatidylcholine (PC ae)\nβ = {bet(cv.loc["Total PC ae","beta"])}, '
                  f'$P$ = {sci(cv.loc["Total PC ae","p"])}', '#E7F5EF', GREEN)
nd(56, 50, 40, 13, f'Diacyl phosphatidylcholine (PC aa)\nβ = {bet(cv.loc["Total PC aa","beta"])}, '
                   f'$P$ = {sci(cv.loc["Total PC aa","p"])}', '#F0EEF6', PURPLE)
arr(24, 37, 42, 27); arr(76, 37, 58, 27)
nd(26, 27, 48, 13, f'PC ae / PC aa ratio\nβ = {bet(cv.loc["PC ae / PC aa","beta"])}, '
                   f'$P$ = {sci(cv.loc["PC ae / PC aa","p"])}', '#EAF2FA', BLUE, fs=6.8)
a.text(50, 7, f'The pre-specified microbiota score and the predicted bacterial fatty-acid pathway both move the ether branch\nand not the ester branch (PC aa $P$ = {sci(nv("pwy5971_total_pc_aa_p"))} for the pathway); n = {n743}.',
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
b.set_xlabel('Effect of the pre-specified score (SD per SD)'); b.set_xlim(-0.16, 0.36)
b.set_title(f'Class-level specificity (n = {n743})', fontsize=8, color=INK, pad=4)
clean(b)

# ------------------------------- C -------------------------------
c = fig.add_subplot(gs[1, 1]); pan(c, 'C')
d1 = SPC[(SPC.group == 'PC ae / total lysoPC') & (SPC.score == 'Pre-specified score')].copy()
cn = d1.species.str.extract(r'C(\d+):(\d+)').astype(float)
d1['carbon'] = cn[0].values; d1['db'] = cn[1].values
c.scatter(d1.carbon + np.random.default_rng(2).normal(0, 0.12, len(d1)), d1.beta, s=16,
          c=[GREEN if q < 0.05 else 'white' for q in d1.q], edgecolors=GREEN, linewidths=0.8)
z = np.polyfit(d1.carbon, d1.beta, 1); xx = np.linspace(d1.carbon.min(), d1.carbon.max(), 20)
c.plot(xx, np.polyval(z, xx), color=ORANGE, lw=1.2)
c.axhline(0, color=GRAY, lw=0.8, ls='--')
c.set_xlabel('PC ae carbon number'); c.set_ylabel('Effect (SD per SD)')
c.set_title('Chain-length gradient, no unsaturation gradient', fontsize=8, color=INK, pad=4)
c.text(0.03, 0.05, f'carbon number $P$ = {nv("carbon_pre_p"):.3f} (CV panel {nv("carbon_cv_p"):.3f})\n'
                   f'double bonds $P$ = {nv("doublebond_pre_p"):.2f}   n = {n743}',
       transform=c.transAxes, fontsize=6.2, color=INK, va='bottom')
clean(c)

plt.savefig(f'Figure3_mech_{ST}.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('chk/F3mech_v65.png', dpi=170, bbox_inches='tight', pad_inches=0.10)
plt.close()
print('Figure 3 (A–C) done')
