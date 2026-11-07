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

EXP = ['Microbiota panel', 'Gemmiger', 'Ether-lipid ratio', 'CKM stage 3 or 4']
npath = ni('n_metacyc_pathways')

fig = plt.figure(figsize=(7.2, 9.4))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.45, 1.95], hspace=0.40, wspace=0.34,
                      left=0.10, right=0.98, top=0.955, bottom=0.055)

# ---------------- A ----------------
a = fig.add_subplot(gs[0, 0]); pan(a, 'A', -0.22)
before = [int((M[(M.exposure == e) & (M.q < 0.05)]).shape[0]) for e in EXP]
after = [int((M[(M.exposure == e) & (M.q < 0.05) & (M.q_nsti_adj < 0.05)]).shape[0]) for e in EXP]
x = np.arange(len(EXP))
a.bar(x - 0.19, before, 0.36, color=ORANGE, label='as reported')
a.bar(x + 0.19, after, 0.36, color=BLUE, label='adjusted for NSTI')
for xi, (b_, af) in enumerate(zip(before, after)):
    a.text(xi - 0.19, b_ + 5, str(b_), ha='center', va='bottom', fontsize=6.2, color=INK)
    a.text(xi + 0.19, af + 5, str(af), ha='center', va='bottom', fontsize=6.2, color=INK)
a.set_xticks(x); a.set_xticklabels(['Microbiota\npanel', '$Gemmiger$', 'Ether-lipid\nratio',
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
c.set_title(f'What survives: the {len(M[(M.exposure=="Microbiota panel")&(M.q<0.05)&(M.q_nsti_adj<0.05)])} '
            f'pathways that remain after NSTI adjustment (9 most negative and 9 most positive shown)',
            fontsize=7.6, color=INK, pad=4)
c.text(0.985, 0.045, 'orange outline: palmitate biosynthesis II — the only pathway that also\n'
                     'survives for the host ether-lipid ratio ($q$* = '
                     f'{nv("pwy5971_r1_q_nsti"):.3f})',
       transform=c.transAxes, ha='right', va='bottom', fontsize=6.0, color=INK, style='italic')
clean(c)

# ---------------- D ----------------
e = fig.add_subplot(gs[2, :]); pan(e, 'D', -0.105)
R_ = REP.sort_values('beta').reset_index(drop=True)
y = np.arange(len(R_))
cols = [ORANGE if q_ < 0.05 else (INK if p_ < 0.05 else GRAY)
        for q_, p_ in zip(R_.q_nsti_adj, R_.p)]
e.errorbar(R_.beta, y, xerr=1.96 * R_.se, fmt='none', lw=0.8, ecolor=GRID, zorder=1)
e.scatter(R_.beta, y, s=[26 if q_ < 0.05 else 15 for q_ in R_.q_nsti_adj],
          c=cols, zorder=3, edgecolors='none')
e.axvline(0, color=GRAY, lw=0.8, ls='--')
e.set_yticks(y); e.set_yticklabels([short(n, 58) for n in R_.name], fontsize=5.8)
e.set_ylim(-0.8, len(R_) - 0.2)
e.set_xlabel('Effect of $Gemmiger$ on the pathway (SD per SD, 95% CI)')
e.set_title(f'Directional replication of the pathways the earlier cohort paper reports for '
            f'$Gemmiger$', fontsize=7.6, color=INK, pad=4)
e.text(0.985, 0.06, f'{ni("rep_n_same_direction")} of {ni("rep_n_pathways")} point the same way '
                    f'as reported there (binomial $P$ = {qfmt(nv("rep_binom_p"))})' +
                    f'\n{ni("rep_n_p05")} nominally significant · '
                    f'{ni("rep_n_survive_nsti")} survive NSTI adjustment (orange)',
       transform=e.transAxes, ha='right', va='bottom', fontsize=6.2, color=INK)
clean(e)
plt.savefig(f'FigureS5_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()
print('Figure done')
