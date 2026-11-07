"""Figure for the MetaCyc enrichment — 20260911_v55.

Four panels, in the order the argument runs:
  A  how many pathways each exposure "finds", before and after adjusting for reference-genome
     coverage (weighted nearest-sequenced taxon index)
  B  why so many are lost: the per-pathway effect of the microbiota panel is nearly collinear
     with the per-pathway effect of NSTI itself
  C  what survives for the microbiota panel, named — the honest version of an enrichment plot
  D  the pathways the earlier cohort paper reported for Gemmiger, tested as a pre-specified
     set: a directional replication
"""
import warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'axes.linewidth': 0.7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.labelsize': 7.5})
BLUE, ORANGE, GREEN, PURPLE = '#0F6CBD', '#D95F02', '#1B9E77', '#7570B3'
GRAY, INK, MUTED, GRID = '#8A8F98', '#2F343A', '#6B7178', '#E3E5E8'
ST = '20260911_v55'
M = pd.read_csv(f'TableS17_metacyc_enrichment_{ST}.csv')
NR = pd.read_csv(f'TableS18_metacyc_vs_nsti_{ST}.csv')
REP = pd.read_csv(f'TableS19_replication_prior_paper_{ST}.csv')
N = pd.read_csv(f'numbers_metacyc_{ST}.csv').set_index('key').value
nv = lambda k: float(N[k]); ni = lambda k: int(float(N[k]))
def clean(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(length=2.5, colors=MUTED)
    ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
def pan(ax, l, x, y=1.03):
    ax.text(x, y, l, transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='left', color=INK)
def qfmt(v):
    if v >= 0.01: return f'{v:.3g}'
    e = int(np.floor(np.log10(v))); m = v / 10 ** e
    return f'{m:.0f}×10{str(e).translate(str.maketrans("0123456789-", chr(0x2070)+chr(0xB9)+chr(0xB2)+chr(0xB3)+"⁴⁵⁶⁷⁸⁹⁻"))}'
def short(s, k=52):
    s = s.replace('&beta;', 'β').replace('&alpha;', 'α')
    return s if len(s) <= k else s[:k - 1].rstrip() + '…'

EXP = ['Microbiota panel', 'Ether-lipid ratio', 'CKM stage 3 or 4']
npath = ni('n_metacyc_pathways')

fig = plt.figure(figsize=(7.2, 9.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.45, 1.30], hspace=0.42,
                      wspace=0.34, left=0.10, right=0.98, top=0.955, bottom=0.055)

# ---------------- A ----------------
a = fig.add_subplot(gs[0, 0]); pan(a, 'A', -0.22)
# 20261030_v65（外審 issue 04）：原本只畫兩根柱，而且把「兩個模型都顯著」的交集
# 標成 adjusted for NSTI，讀起來像是校正模型自己的發現數。三個數字分開畫。
before = [int((M[(M.exposure == e) & (M.q < 0.05)]).shape[0]) for e in EXP]
adjmod = [int((M[(M.exposure == e) & (M.q_nsti_adj < 0.05)]).shape[0]) for e in EXP]
after = [int((M[(M.exposure == e) & (M.q < 0.05) & (M.q_nsti_adj < 0.05)]).shape[0]) for e in EXP]
x = np.arange(len(EXP))
a.bar(x - 0.26, before, 0.25, color=ORANGE, label='primary model')
a.bar(x + 0.00, adjmod, 0.25, color='#8FB8D8', label='NSTI-adjusted model')
a.bar(x + 0.26, after, 0.25, color=BLUE, label='significant in both')
for xi in range(len(EXP)):
    for dx, v in ((-0.26, before[xi]), (0.0, adjmod[xi]), (0.26, after[xi])):
        a.text(xi + dx, v + 5, str(v), ha='center', va='bottom', fontsize=5.8, color=INK)
a.set_xticks(x); a.set_xticklabels(['Microbiota\npanel', 'Ether-lipid\nratio',
                                    'CKM stage\n3 or 4'], fontsize=6.4)
a.set_ylabel(f'Pathways at $q$ < 0.05\n(of {npath} tested)')
a.set_ylim(0, max(before) * 1.20)
a.legend(frameon=False, fontsize=6.2, loc='upper right')
a.set_title(f'Most "enriched" pathways do not survive\nadjustment for reference coverage '
            f'(n = {ni("n_picrust_samples")})', fontsize=7.6, color=INK, pad=4)
clean(a)

# ---------------- B ----------------
b = fig.add_subplot(gs[0, 1]); pan(b, 'B', -0.22)
d = M[M.exposure == 'Microbiota panel'].merge(NR[['pathway', 'beta']], on='pathway',
                                              suffixes=('', '_nsti'))
r = np.corrcoef(d.beta, d.beta_nsti)[0, 1]
b.scatter(d.beta_nsti, d.beta, s=9, color=GRAY, alpha=0.5, edgecolors='none')
sig = d[d.q < 0.05]
b.scatter(sig.beta_nsti, sig.beta, s=11, color=ORANGE, alpha=0.85, edgecolors='none')
lim = np.array([-0.5, 0.5]); b.plot(lim, lim, ls='--', lw=0.9, color=GRID)
b.plot(lim, np.polyval(np.polyfit(d.beta_nsti, d.beta, 1), lim), color=BLUE, lw=1.2)
b.axhline(0, color=GRID, lw=0.7); b.axvline(0, color=GRID, lw=0.7)
b.set_xlim(-0.5, 0.5); b.set_ylim(-0.5, 0.5)
b.set_xlabel('Effect of NSTI on the pathway (SD per SD)')
b.set_ylabel('Effect of the microbiota panel\n(SD per SD)')
b.set_title('The two effects are nearly the same thing', fontsize=7.6, color=INK, pad=4)
b.text(0.04, 0.96, f'$r$ = {r:+.3f} across all {npath} pathways\n'
                   f'{ni("metacyc_nsti_n_q05")} of {npath} track NSTI itself',
       transform=b.transAxes, fontsize=6.2, color=INK, va='top')
clean(b)

# ---------------- C ----------------
c = fig.add_subplot(gs[1, :]); pan(c, 'C', -0.105)
s = M[(M.exposure == 'Microbiota panel') & (M.q < 0.05) &
      (M.q_nsti_adj < 0.05)].sort_values('beta')
s = pd.concat([s.head(9), s.tail(9)])
y = np.arange(len(s))
cols = [GREEN if v > 0 else PURPLE for v in s.beta]
c.barh(y, s.beta, 0.68, color=cols,
       edgecolor=[ORANGE if p_ == 'PWY-5971' else 'none' for p_ in s.pathway],
       linewidth=[1.6 if p_ == 'PWY-5971' else 0 for p_ in s.pathway])
c.axvline(0, color=GRAY, lw=0.8)
c.set_yticks(y)
c.set_yticklabels([short(n) for n in s.name], fontsize=6.0)
for yy, (_, r_) in zip(y, s.iterrows()):
    off = 0.012 if r_.beta > 0 else -0.012
    c.text(r_.beta + off, yy, f'$q$* = {qfmt(r_.q_nsti_adj)}',
           va='center', ha='left' if r_.beta > 0 else 'right', fontsize=5.4, color=MUTED)
c.set_xlim(-0.46, 0.62)
c.set_xlabel('Effect of the microbiota panel on the pathway (SD per SD)')
c.set_title(f'The {len(M[(M.exposure=="Microbiota panel")&(M.q<0.05)&(M.q_nsti_adj<0.05)])} pathways '
            f'significant in both models (9 most negative and 9 most positive shown)',
            fontsize=7.6, color=INK, pad=4)
c.text(0.985, 0.045, 'orange outline: palmitate biosynthesis II — the only pathway that also\n'
                     'survives for the host ether-lipid ratio ($q$* = '
                     f'{nv("pwy5971_r1_q_nsti"):.3f})',
       transform=c.transAxes, ha='right', va='bottom', fontsize=6.0, color=INK, style='italic')
clean(c)


# ---------------- E：陰性臂（原 Figure 3D 右半） ----------------
gse = gs[2, :].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.06)
f = fig.add_subplot(gse[0, 0]); pan(f, 'D', -0.215)
P = M[M.exposure == 'Microbiota panel'].set_index('pathway')
GEM = set(REP.pathway)
NEG = ['P162-PWY', 'PWY-5022', 'HISDEG-PWY', 'PWY-7254', 'P108-PWY', 'PWY-4984']
for pw in NEG:
    r_ = P.loc[pw]
    assert r_.q < 0.05 and r_.q_nsti_adj < 0.05 and r_.beta < 0, f'{pw} 不符陰性臂條件'
s_ = P.loc[NEG].sort_values('beta', ascending=False)
yy = np.arange(len(s_))
f.barh(yy, s_.beta, 0.34, color=PURPLE, edgecolor='white', linewidth=0.6)
f.axvline(0, color=GRAY, lw=0.8)
for i, (pw, r_) in enumerate(s_.iterrows()):
    nm = r_['name'].replace('&beta;', 'β')
    if not nm.startswith('L-'):
        nm = nm[0].upper() + nm[1:]
    pass  # 20261030_v65：不再標 †（不逐條對照人家的圖）
    f.text(-0.455, i + 0.30, nm, va='bottom', ha='left', fontsize=5.8, color=INK)
    f.text(r_.beta - 0.008, i, f'$q$* = {qfmt(r_.q_nsti_adj)}', va='center', ha='right',
           fontsize=5.4, color=MUTED)
f.set_yticks([]); f.set_ylim(-0.7, len(s_) - 0.25)
f.set_xlim(-0.46, 0.02); f.set_xticks([-0.4, -0.3, -0.2, -0.1, 0.0])
f.set_xlabel('Effect of the microbiota panel on the pathway (SD per SD)', fontsize=6.6)
f.set_title('Runs the other way: protein and organic-acid degradation',
            fontsize=7.6, color=PURPLE, pad=5)
f.spines[['top', 'right', 'left']].set_visible(False)
f.tick_params(length=2.5, colors=MUTED, labelsize=6.4)
f.xaxis.label.set_color(INK)
n_neg_all = int(((P.q < 0.05) & (P.q_nsti_adj < 0.05) & (P.beta < 0)).sum())
n_pos_all = int(((P.q < 0.05) & (P.q_nsti_adj < 0.05) & (P.beta > 0)).sum())
# † 只有在真的有畫出 † 時才解釋，否則留下一行沒有指涉對象的註腳
dag_note = ''
g = fig.add_subplot(gse[0, 1]); g.axis('off')
g.text(0.0, 0.97,
       'The carbohydrate-fermentation arm in panel C runs one way and these six run the\n'
       'other, so the panel is not a single gradient of "more bacteria, more pathways".\n'
       f'Of the {n_neg_all + n_pos_all} pathways surviving adjustment for reference-genome coverage, '
       f'{n_pos_all} rise\nwith the microbiota panel and {n_neg_all} fall; the six drawn here are the '
       'named\nprotein- and organic-acid-degradation routes that fall, three of which\n'
       '(P162-PWY, PWY-5022, HISDEG-PWY) also appear among the nine most negative\n'
       'pathways in panel C while the other three do not.\n\n'
       'β is from the primary model and $q$* from the model additionally adjusted for\n'
       f'reference-genome coverage; every pathway shown reaches $q$ < 0.05 in both.\n'
       + dag_note +
       f'Predicted metagenome (PICRUSt2), n = {int(P.n.iloc[0]):,}.',
       transform=g.transAxes, fontsize=6.2, color=MUTED, va='top', ha='left', linespacing=1.75)

plt.savefig('FigureS5_20261030_v65.svg', bbox_inches='tight', pad_inches=0.10)
plt.savefig('chk/S5_v65.png', dpi=170, bbox_inches='tight', pad_inches=0.10)
plt.close()
print('Figure S5: A three bars, B, C, D negative arm (Gemmiger replication removed)')
