"""Graphical abstract, v65 — 20261030.

改以事前三菌屬分數為主要暴露；臨床端改成「鏈走到哪一刀」：CKM ≥2 到得了、≥3 到不了；
中介改報八條平行中介在 ≥2 那一刀的 63%。

Two gaps rather than three: the genotype band is gone with the rest of the genetics. The
chain keeps its four links, and the strip beneath it carries the two results that say what
kind of signal this is — distributed across the community, not a single organism.
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED = '#8A8F98', '#2F343A', '#6B7178'
ST = '20261030_v65'
N = pd.concat([pd.read_csv(f'numbers_v65_all_{ST}.csv'), pd.read_csv(f'numbers_extra_{ST}.csv')]).drop_duplicates('key', keep='last').set_index('key').value
nv = lambda k: float(N[k]); ni = lambda k: int(float(N[k]))
def sci(p, d=1):
    if p >= 0.01: return f'{p:.3g}'
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f'{m:.{d}f} × 10$^{{{e}}}$'
S8 = pd.read_csv(f'out/TableS8_ratio_outcome_{ST}.csv')

fig = plt.figure(figsize=(7.2, 5.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
def box(x, y, w, h, fc, ec, lw=1.0, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.4,rounding_size={r}',
                                fc=fc, ec=ec, lw=lw, zorder=1))
def txt(x, y, s, fs=7, c=INK, ha='center', va='center', w='normal', st='normal', ls=1.35):
    ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, fontweight=w, style=st,
            linespacing=ls, zorder=3)
def arrow(x1, y1, x2, y2, c=INK, lw=2.0, ms=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle=f'-|>,head_width={ms/28},head_length={ms/18}',
                                 lw=lw, color=c, zorder=2, mutation_scale=ms))

# ---------------- band 1: the gap ----------------
box(2, 79, 96, 18, '#FBF7F2', '#C9A227', 1.1)
txt(5, 93.8, 'THE CLINICAL GAP', 7.8, '#8A6D1A', ha='left', w='bold')
gaps = [('CKM staging tells you WHERE a patient is,\nnot WHY they got there',
         'nothing refines risk within a stage'),
        ('The microbiome is modifiable but there is\nno defined host readout to target',
         'interventions have no endpoint')]
for i, (a_, b_) in enumerate(gaps):
    x = 7 + i * 47
    txt(x, 90.0, f'{i+1}', 11, '#C9A227', ha='left', va='top', w='bold')
    txt(x + 21, 87.0, a_, 6.7, INK, ha='center')
    txt(x + 21, 81.8, b_, 5.9, MUTED, ha='center', st='italic')
    if i < 1: ax.plot([x + 44, x + 44], [81.0, 91.0], color='#E6DCC4', lw=0.9, zorder=0)

# ---------------- band 2: the chain ----------------
box(2, 24, 96, 52, '#F7FAFC', '#CBD5E0', 1.0)
txt(5, 72.5, 'WHAT WE FOUND', 7.8, BLUE, ha='left', w='bold')
txt(96, 72.5, f'{ni("n_microbiome_all")} adults across the CKM spectrum  ·  '
              f'gut microbiome + metabolome', 6.2, MUTED, ha='right')
cy = 53
box(4, cy - 8.0, 17, 16, '#E7F5EF', GREEN, 1.4)
txt(12.5, cy + 4.4, 'GUT MICROBIOTA', 6.6, '#14634A', w='bold')
txt(12.5, cy - 0.2, 'pre-specified\n3-genus score', 6.0, INK)
txt(12.5, cy - 5.6, 'modifiable', 5.8, GREEN, st='italic', w='bold')
arrow(21.8, cy, 28.2, cy, GREEN, 2.2, 11)
txt(25, cy + 7.2, f'β = {nv("rob_pre_base_beta"):+.2f}\n$P$ = {sci(nv("rob_pre_base_p"))}'.replace(' × 10', '\n×10'),
    5.6, '#14634A', w='bold')
box(29, cy - 10.0, 20, 20, '#FDF6E3', '#B7791F', 1.4)
txt(39, cy + 7.2, 'ONE BIOCHEMICAL STEP', 6.2, '#8A6D1A', w='bold')
txt(39, cy + 2.2, 'sn-1 ether-bond\nformation\n(peroxisomal)', 6.4, INK)
txt(39, cy - 4.8, f'PC ae  ↑   $P$ = {nv("cls_pre_total_pc_ae_p"):.4f}\n'
                  f'PC aa  —   $P$ = {nv("cls_pre_total_pc_aa_p"):.2f}', 5.9, INK, w='bold')
txt(39, cy - 8.6, f'ratio $P$ = {sci(nv("cls_pre_pc_ae_over_pc_aa_p"))}', 5.7, '#8A6D1A', w='bold')
arrow(49.6, cy, 56.4, cy, '#B7791F', 2.2, 11)
box(57, cy - 8.0, 17, 16, '#EAF2FA', BLUE, 1.4)
txt(65.5, cy + 4.4, 'PLASMA ETHER', 6.6, '#0A4E8A', w='bold')
txt(65.5, cy + 1.8, 'LIPID POOL', 6.6, '#0A4E8A', w='bold')
txt(65.5, cy - 3.4, f'{ni("spc_pre_q05")} of {ni("spc_pre_n")} species\n'
                    f'100% same direction', 6.0, INK)
arrow(74.6, cy, 81.4, cy, BLUE, 2.2, 11)
r1 = S8[S8.ratio == 'R1'].set_index('outcome')
box(82, cy - 10.0, 15, 20, '#FDECEC', '#C05621', 1.4)
txt(89.5, cy + 7.0, 'CKM STAGE', 6.6, '#8C3D12', w='bold')
txt(89.5, cy + 2.8, f'0–4: β = {r1.loc["CKM stage (5-level)","beta"]:.3f}\n$q$ = {sci(r1.loc["CKM stage (5-level)","q"], 0)}'.replace('-', '−'), 5.8, INK)
txt(89.5, cy - 2.6, f'stage ≥ 2\n$P$ = {sci(nv("bnd_r1_ge2_p"), 0)}', 5.8, '#8C3D12', w='bold')
txt(89.5, cy - 7.4, f'stage ≥ 3\n$P$ = {nv("bnd_r1_ge3_p"):.2f}  (none)', 5.6, MUTED)
txt(50, 34.0, f'No detectable direct microbiota effect on any stage boundary: {round(100 * nv("pm_pre_ge2_prop_mediated"))}% of its '
              f'association with CKM stage ≥ 2 runs through the eight ratios', 6.1, BLUE, w='bold')
txt(50, 27.6, f'The chain reaches the metabolic boundary (stage ≥ 2) and stops there: stage ≥ 3 and ≥ 4 are carried by '
              f'ratios the microbiota does not move\neight ratios add ΔAUC {nv("auc8_dm_eight_delta"):+.2f} for diabetes, '
              f'{nv("auc8_tg_eight_delta"):+.2f} for triglycerides, {nv("auc8_ckm34_eight_delta"):+.2f} for stage 3–4', 5.6, MUTED, st='italic')

# ---------------- band 3: what this adds ----------------
box(2, 2, 96, 19, '#F0FBF6', GREEN, 1.3)
txt(5, 17.5, 'WHAT THIS ADDS', 7.8, '#14634A', ha='left', w='bold')
txt(50, 12.2, 'A modifiable input, a measurable intermediate, and the stage at which to use it.',
    9.0, '#14634A', w='bold')
txt(50, 6.4, 'A plasma ether-lipid ratio points at a named enzymatic step, is attached to an exposure that can be changed,\n'
             'and carries the microbiota\'s association with CKM at the metabolic stage — giving microbiome-directed\n'
             'intervention a biochemical endpoint, and CKM stage 1–2 as the window in which to apply it.', 6.6, INK)
plt.savefig(f'GraphicalAbstract_{ST}.svg', bbox_inches='tight'); plt.savefig('chk/GA_v65.png', dpi=170, bbox_inches='tight'); plt.close()
print('Graphical abstract done')
