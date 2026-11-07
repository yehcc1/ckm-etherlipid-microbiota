"""Metabolite-level route map (the counterpart of Figure 4c of Wu et al. 2025) — 20261031_v65.

Draws the predicted microbial route from dietary sucrose to bacterial palmitate and on to the host
sn-1 ether bond at the level of metabolites and enzymes, each MetaCyc pathway as a coloured region;
and, on the right, the opposite arm (protein and organic-acid degradation), which falls with the
pre-specified score and rises with Enterocloster.

Every number printed is read from TableSmc_metacyc_taxa_20261031_v65.csv (X28): β from the primary
model, q* from the model additionally adjusted for reference-genome coverage.  A region is drawn
solid when the pathway survives (q < 0.05 in both models) for the pre-specified score and dashed
when it survives only for one of the three pre-specified genera; the script asserts the sign and
survival that the layout claims, so a layout that disagrees with the table cannot be drawn.  Chips
name every pre-specified genus whose own model survives, with the sign of that genus' coefficient.
Intermediates follow the MetaCyc pathway pages (PWY-5384, PWY-621, P122-PWY, P124-PWY,
FERMENTATION-PWY, P125-PWY, PWY-6588, PWY-5971, HISDEG-PWY, P162-PWY, PWY-5022, PWY-7254, P108-PWY).
"""
import numpy as np, pandas as pd
from matplotlib.patches import FancyBboxPatch

GREEN, PURPLE, BLUE, GRAY, INK, MUTED = '#1B9E77', '#7570B3', '#0F6CBD', '#8A8F98', '#2F343A', '#6B7178'
LINK = '#B9BDC4'
GENUS = {'NK4A136': ('#D95F02', 'Lachnospiraceae NK4A136 group'),
         'R-7': ('#0F6CBD', 'Christensenellaceae R-7 group'),
         'Enterocloster': ('#B8921E', 'Enterocloster')}
FILL = {+1: '#E7F5EF', -1: '#EFEEF7'}
EDGE = {+1: GREEN, -1: PURPLE}
HUB = dict(fc='#F3F4F6', ec=GRAY)


def qf(q):
    if q >= 0.01: return f'{q:.3g}'
    e = int(np.floor(np.log10(q))); m = q / 10 ** e
    return f'{m:.1f} × 10$^{{{e}}}$'


def bet(v):
    return f'{v:+.2f}'.replace('-', '−')


class Map:
    def __init__(self, ax, table, fs=5.2):
        self.ax = ax; self.fs = fs
        M = pd.read_csv(table)
        self.P = M[M.exposure == 'Pre-specified score'].set_index('pathway')
        self.G = {k: M[M.exposure == v[1]].set_index('pathway') for k, v in GENUS.items()}
        self.R1 = M[M.exposure == 'Ether-lipid ratio'].set_index('pathway')
        self.n = int(self.P.n.iloc[0])
        self.used = []
        self.ax.figure.canvas.draw()
        self.rend = self.ax.figure.canvas.get_renderer()

    def _w(self, t):
        """width of a text artist in data units"""
        bb = t.get_window_extent(self.rend)
        inv = self.ax.transData.inverted()
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return x1 - x0

    # ---- primitives ------------------------------------------------------------------
    def met(self, x, y, txt, bold=False, fc='white', ec='#C9CDD3', fs=None, z=5, ha='center'):
        return self.ax.text(x, y, txt, ha=ha, va='center', fontsize=fs or self.fs, color=INK,
                            fontweight='bold' if bold else 'normal', zorder=z, linespacing=1.15,
                            bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.5', fc=fc, ec=ec, lw=0.6))

    def arrow(self, x1, y1, x2, y2, enz=None, col=GRAY, lw=0.8, sa=6, sb=6, side=+1, fs=None, dx=0, dy=0, head=True):
        self.ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=4,
                         arrowprops=dict(arrowstyle='-|>' if head else '-', lw=lw, color=col, mutation_scale=7,
                                         shrinkA=sa, shrinkB=sb))
        if enz:
            mx, my = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
            if abs(x2 - x1) < abs(y2 - y1):   # vertical: label to the side
                self.ax.text(mx + 1.0 * side, my, enz, ha='left' if side > 0 else 'right', va='center',
                             fontsize=fs or self.fs - 1.0, color=MUTED, style='italic', zorder=6, linespacing=1.1)
            else:
                self.ax.text(mx, my + 1.2 * side, enz, ha='center', va='bottom' if side > 0 else 'top',
                             fontsize=fs or self.fs - 1.0, color=MUTED, style='italic', zorder=6, linespacing=1.1)

    def elbow(self, pts, enz=None, col=GRAY, lw=0.8, sa=6, sb=6, side=+1, fs=None, dx=0, dy=0, lab_seg=0):
        for i in range(len(pts) - 1):
            last = i == len(pts) - 2
            self.arrow(*pts[i], *pts[i + 1], col=col, lw=lw, sa=sa if i == 0 else 0, sb=sb if last else 0,
                       head=last, enz=enz if i == lab_seg else None, side=side, fs=fs, dx=dx, dy=dy)

    def chips(self, x, y, items, maxw, fs):
        """row(s) of genus chips starting at (x, y) going down; wraps at maxw.  Returns the y below."""
        cx, cy, lh = x, y, 2.7
        for g, b in items:
            t = self.ax.text(cx, cy, f'{g} {bet(b)}', ha='left', va='top', fontsize=fs, color='white',
                             fontweight='bold', zorder=7,
                             bbox=dict(boxstyle='round,pad=0.2,rounding_size=0.4', fc=GENUS[g][0], ec='none'))
            w = self._w(t) + 1.2
            if cx + w > x + maxw and cx > x:       # wrap
                t.set_position((x, cy - lh)); cy -= lh; cx = x + w
            else:
                cx += w
        return cy - lh

    def region(self, x, y, w, h, title, pw, expect, tx=None, ty=None, bw=None, fs=None, note=None):
        """Coloured pathway region with a title block (title, statistic, chips) anchored at (tx, ty)."""
        r = self.P.loc[pw]
        assert np.sign(r.beta) == expect, f'{pw} sign {np.sign(r.beta)} is not {expect}'
        ch = [(g, self.G[g].loc[pw, 'beta']) for g in GENUS if self.G[g].loc[pw, 'survives']]
        assert r.survives or any(np.sign(b) == expect for _, b in ch), f'{pw}: nothing survives'
        self.used.append(pw)
        fs = fs or self.fs
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                         fc=FILL[expect], ec=EDGE[expect], lw=1.0 if r.survives else 0.9,
                                         ls='-' if r.survives else (0, (2.4, 1.6)), zorder=1))
        tx = x + 1.2 if tx is None else tx; ty = y + h - 1.0 if ty is None else ty; bw = bw or (w - 2.4)
        self.ax.text(tx, ty, title, ha='left', va='top', fontsize=fs + 0.5, fontweight='bold',
                     color=EDGE[expect], zorder=6, linespacing=1.15)
        nl = title.count('\n') + 1
        yy = ty - 2.9 * nl - 0.2
        stat = (f'score β {bet(r.beta)}, $q$* {qf(r.q_nsti_adj)}' if r.survives
                else f'score $q$* {qf(r.q_nsti_adj)} (n.s.)')
        self.ax.text(tx, yy, stat, ha='left', va='top', fontsize=fs - 0.3,
                     color=INK if r.survives else MUTED, zorder=6)
        yy -= 2.9
        yy = self.chips(tx, yy, ch, bw, fs - 0.8)
        if note:
            self.ax.text(tx, yy + 0.3, note, ha='left', va='top', fontsize=fs - 1.0, color=MUTED,
                         style='italic', zorder=6, linespacing=1.15)
        return r

    def plain_region(self, x, y, w, h, label, ec=GRAY, fc='#F7F8F9'):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                         fc=fc, ec=ec, lw=0.8, ls=(0, (2.4, 1.6)), zorder=1))
        self.ax.text(x + 1.2, y + h - 1.0, label, ha='left', va='top', fontsize=self.fs + 0.2,
                     style='italic', color=MUTED, zorder=6)


def draw(ax, table, host, fs=5.2):
    """host: dict with keys beta (float), p (formatted string), n (int) for PWY-5971 → ether-lipid ratio."""
    m = Map(ax, table, fs=fs)
    ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(-31, 106)
    met, arrow, region, elbow = m.met, m.arrow, m.region, m.elbow
    X0, X1 = 0.5, 60      # positive arm
    R0, R1 = 63.5, 99.5   # opposite arm

    # ================================ positive arm ================================
    ax.text(X0, 100.6, 'Rises with the pre-specified score:\ncarbohydrate fermentation → bacterial fatty acid → host ether bond',
            fontsize=fs + 1.2, color=GREEN, fontweight='bold', ha='left', va='bottom', linespacing=1.15)
    met(29, 95, 'dietary sucrose', bold=True, **HUB, fs=fs + 0.4)

    # sucrose degradation IV (phosphorylase): title block left, node right
    region(X0, 79.5, 25.5, 11.5, 'Sucrose degradation IV', 'PWY-5384', +1, bw=15.5)
    met(21.5, 85.5, 'α-D-glucose 1-P\n+ D-fructose')
    arrow(26.5, 93.5, 22.5, 88.3, enz='sucrose\nphosphorylase', side=-1, sa=6, sb=3, dx=1.0, dy=0.3)
    arrow(21.5, 83, 26.5, 74, enz='phosphogluco-\nmutase', side=-1, sa=5, sb=5, dx=-0.5)

    # sucrose degradation III (invertase): node left, title block right
    region(33.5, 79.5, 26.5, 11.5, 'Sucrose degradation III', 'PWY-621', +1, tx=43, bw=16)
    met(38, 85.5, 'D-glucose\n+ D-fructose')
    arrow(31.5, 93.5, 37, 88.3, enz='invertase', side=+1, sa=6, sb=3, dx=1.2, dy=0.6)
    arrow(38, 83, 32, 74, enz='hexokinase', side=+1, sa=5, sb=5, dx=0.8)

    # hexose-phosphate hub
    met(29, 72, 'glucose 6-P  ⇌  fructose 6-P', bold=True, **HUB, fs=fs + 0.2)

    # heterolactic fermentation: title block left column, chain right column
    region(X0, 44.5, 20.5, 24.5, 'Heterolactic\nfermentation', 'P122-PWY', +1, tx=10.5, ty=65, bw=10)
    xc = 5.2
    met(xc, 64, '6-phospho-\nD-gluconate')
    met(xc, 56, 'D-xylulose 5-P')
    met(xc, 47.5, 'acetyl-P\n+ GAP')
    arrow(23, 71.2, xc + 1.5, 66.5, sa=7, sb=3)
    arrow(xc, 62, xc, 57.5, sa=3, sb=3)
    arrow(xc, 54.5, xc, 49.5, enz='phospho-\nketolase', side=+1, sa=3, sb=3)

    # Bifidobacterium shunt: chain left column, title block right column
    region(39.5, 44.5, 20.5, 24.5, 'Bifidobacterium\nshunt', 'P124-PWY', +1, tx=40.7, ty=65, bw=10)
    xb = 55
    met(xb, 64, 'acetyl-P +\nD-erythrose 4-P')
    met(xb, 56, 'D-xylulose 5-P')
    met(xb, 47.5, 'acetyl-P\n+ GAP')
    arrow(35, 71.2, xb - 1.5, 66.5, enz='F6P phospho-\nketolase', side=+1, sa=7, sb=3, dx=-1.0, dy=1.2)
    arrow(xb, 62, xb, 57.5, sa=3, sb=3)
    arrow(xb, 54.5, xb, 49.5, enz='X5P phospho-\nketolase', side=-1, sa=3, sb=3)

    # glycolysis (no statistic)
    m.plain_region(23, 44.5, 14, 24.5, 'glycolysis')
    met(29, 62, 'fructose\n1,6-bisP', fs=fs - 0.3)
    met(29, 55.5, 'GAP', fs=fs - 0.3)
    met(29, 49.5, 'PEP', fs=fs - 0.3)
    arrow(29, 70.5, 29, 64.5, sa=5, sb=3)
    arrow(29, 60, 29, 57, sa=3, sb=3); arrow(29, 54, 29, 51, sa=3, sb=3)
    arrow(29, 48, 29, 41.5, sa=3, sb=5)
    met(29, 39.5, 'pyruvate', bold=True, **HUB, fs=fs + 0.6)

    # acetyl-CoA hub, fed by pyruvate and by both phosphoketolase products
    met(29, 28, 'acetyl-CoA', bold=True, **HUB, fs=fs + 0.6)
    arrow(29, 37.5, 29, 30, enz='PFOR / PFL', side=+1, sa=4, sb=4)
    elbow([(xc, 45), (xc, 28), (25.5, 28)], enz='PTA', side=-1, sa=4, sb=2, lab_seg=0)
    elbow([(xb, 45), (xb, 28), (32.5, 28)], enz='PTA', side=+1, sa=4, sb=2, lab_seg=0)

    # mixed-acid fermentation (from pyruvate, left)
    region(7, 29.5, 14.5, 14.5, 'Mixed acid\nfermentation', 'FERMENTATION-PWY', +1, bw=13.5,
           note='pyruvate → lactate, formate,\nacetate, ethanol, succinate')
    arrow(25, 39.5, 21.5, 38.5, sa=5, sb=2)

    # (R,R)-butanediol (from pyruvate, right)
    region(39, 29.5, 14.5, 14.5, '(R,R)-Butanediol\nbiosynthesis', 'P125-PWY', +1, bw=13.5,
           note='2 pyruvate → 2-acetolactate\n→ acetoin → 2,3-butanediol')
    arrow(33, 39.5, 39, 38.5, sa=5, sb=2)

    # acetone (from acetyl-CoA, left) — drawn via Christensenellaceae R-7
    region(X0, 11.5, 19.5, 14.5, 'Pyruvate fermentation\nto acetone', 'PWY-6588', +1, bw=18,
           note='2 acetyl-CoA → acetoacetyl-CoA\n→ acetoacetate → acetone')
    arrow(25.5, 27, 17, 26, enz='thiolase', side=-1, sa=4, sb=2, dy=-0.2)

    # palmitate biosynthesis II — the end of the route
    r1 = m.R1.loc['PWY-5971']
    assert r1.survives and r1.beta > 0 and m.R1.survives.sum() == 1, 'PWY-5971 is not the sole R1 survivor'
    region(22.5, 1.5, 37.5, 25, 'Palmitate biosynthesis II\n(type II fatty acid synthase)', 'PWY-5971', +1,
           tx=38, bw=21,
           note=f'the only one of 365 pathways that also survives\ncoverage adjustment against the host ratio R1\n'
                f'(β {bet(r1.beta)}, $q$* {qf(r1.q_nsti_adj)}, n = {int(r1.n)})')
    met(29, 20.5, 'malonyl-CoA')
    met(29, 13, 'malonyl-ACP\n→ 3-oxoacyl-ACP\n→ enoyl-ACP → acyl-ACP\n(×7 cycles)', fs=fs - 0.5)
    met(29, 4.5, 'palmitate (C16:0)', bold=True)
    arrow(29, 26, 29, 22.5, enz='acetyl-CoA\ncarboxylase', side=+1, sa=4, sb=3)
    arrow(29, 18.5, 29, 17, enz='FabD, FabH', side=-1, sa=3, sb=3)
    arrow(29, 9.5, 29, 6.5, enz='FabG, FabZ, FabI, FabF', side=-1, sa=3, sb=3)

    # host
    ax.add_patch(FancyBboxPatch((X0, -19.5), X1 - X0, 19.5, boxstyle='round,pad=0.4,rounding_size=1.2',
                                fc='#EAF2FA', ec=BLUE, lw=1.1, zorder=1))
    ax.text(35, -1.2, 'HOST — the sn-1 ether bond', ha='left', va='top', fontsize=fs + 0.5,
            fontweight='bold', color=BLUE, zorder=6)
    ax.text(35, -4.3, f'palmitate biosynthesis II → ether-lipid ratio R1:\nβ = {host["beta"]:+.3f} SD per SD, '
            f'$P$ = {host["p"]}, n = {host["n"]}'.replace('-0', '−0'),
            ha='left', va='top', fontsize=fs - 0.3, color=INK, zorder=6, linespacing=1.2)
    met(8.5, -14, 'hexadecanol', fc='white', ec=BLUE)
    met(27, -14, '1-O-hexadecyl-DHAP', fc='white', ec=BLUE)
    met(48.5, -14, 'PC ae  (sn-1 ether)\nR1 = PC ae / lysoPC', bold=True, fc='white', ec=BLUE, fs=fs - 0.2)
    arrow(29, 2.5, 10, -11.5, col=BLUE, lw=1.1, enz='FAR1', side=+1, sa=5, sb=5, dx=1.5, dy=1.0, fs=fs - 0.6)
    arrow(14.5, -14, 20, -14, col=BLUE, enz='AGPS\n(+ acyl-DHAP from GNPAT)', side=-1, sa=2, sb=2, fs=fs - 1.0)
    arrow(34, -14, 40.5, -14, col=BLUE, enz='ADHAPR, AGPAT,\nLPCAT, CPT', side=-1, sa=2, sb=2, fs=fs - 1.0)

    # ================================ opposite arm ================================
    ax.text(R0, 100.6, 'Falls with the score, rises with $Enterocloster$:\nprotein and organic-acid degradation',
            fontsize=fs + 1.2, color=PURPLE, fontweight='bold', ha='left', va='bottom', linespacing=1.15)
    xs = 91.5   # the chain column
    met(xs, 95, 'L-histidine', bold=True, **HUB, fs=fs + 0.4)
    region(R0, 80, R1 - R0, 11.5, 'L-Histidine\ndegradation I', 'HISDEG-PWY', -1, bw=18)
    met(xs, 88, 'urocanate', fs=fs - 0.2)
    met(xs, 82.7, 'N-formimino-\nL-glutamate', fs=fs - 0.4)
    arrow(xs, 93, xs, 89.5, enz='histidine\nammonia-lyase', side=-1, sa=4, sb=2)
    arrow(xs, 86.5, xs, 84.5, sa=2, sb=2)
    arrow(xs, 80.8, xs, 77.5, enz='formimino-\nglutamase', side=-1, sa=2, sb=4)
    met(xs, 75.8, 'L-glutamate', bold=True, **HUB, fs=fs + 0.4)

    region(R0, 51.5, R1 - R0, 22, 'L-Glutamate\ndegradation V\n(via hydroxyglutarate)', 'P162-PWY', -1, bw=18)
    met(xs, 70.5, '2-oxoglutarate', **HUB)
    met(xs, 65, '(R)-2-hydroxyglutarate', fs=fs - 0.3)
    met(xs, 59.5, 'glutaconyl-CoA', fs=fs - 0.3)
    met(xs, 54, 'crotonyl-CoA →\nbutanoate + acetate', fs=fs - 0.5)
    arrow(xs, 74, xs, 72, enz='GDH', side=-1, sa=4, sb=2)
    arrow(xs, 69, xs, 66.5, sa=2, sb=2); arrow(xs, 63.5, xs, 61, sa=2, sb=2); arrow(xs, 58, xs, 56, sa=2, sb=2)

    region(R0, 29, R1 - R0, 20.5, '4-Aminobutanoate\n(GABA) degradation V', 'PWY-5022', -1, bw=18)
    met(xs, 46.5, '4-aminobutanoate', bold=True, fs=fs - 0.2)
    met(xs, 41, 'succinate\nsemialdehyde', fs=fs - 0.4)
    met(xs, 35.5, '4-hydroxybutanoate\n→ 4-hydroxybutanoyl-CoA', fs=fs - 0.6)
    met(xs, 30.5, 'crotonyl-CoA → butanoate', fs=fs - 0.5)
    arrow(xs, 45, xs, 43, enz='GABA transaminase\n(2-OG → L-glutamate)', side=-1, sa=2, sb=2, fs=fs - 1.3)
    arrow(xs, 39, xs, 37.5, sa=2, sb=2); arrow(xs, 33.5, xs, 32, sa=2, sb=2)

    region(R0, 10, R1 - R0, 17, 'TCA cycle VII\n(acetate-producers)', 'PWY-7254', -1, bw=18)
    met(xs, 24, 'acetyl-CoA +\noxaloacetate', fs=fs - 0.4)
    met(xs, 18, 'citrate → isocitrate\n→ 2-oxoglutarate', fs=fs - 0.5)
    met(xs, 12.5, 'oxaloacetate → malate\n→ fumarate → succinate', fs=fs - 0.5)
    arrow(xs, 22, xs, 20, enz='citrate\nsynthase', side=-1, sa=2, sb=2)
    arrow(xs, 16, xs, 14.5, enz='reductive\nbranch', side=-1, sa=2, sb=2)

    region(R0, -19.5, R1 - R0, 27.5, 'Pyruvate fermentation\nto propanoate I\n(succinate pathway)', 'P108-PWY', -1, bw=18)
    met(xs, 4.5, 'pyruvate → oxaloacetate\n→ malate → fumarate', fs=fs - 0.5)
    met(xs, -1.5, 'succinate', bold=True, **HUB)
    met(xs, -6.8, 'succinyl-CoA →\n(R)-methylmalonyl-CoA', fs=fs - 0.5)
    met(xs, -12.5, 'propanoyl-CoA', fs=fs - 0.3)
    met(xs, -17, 'propanoate', bold=True)
    arrow(xs, 2.5, xs, 0, enz='fumarate\nreductase', side=-1, sa=2, sb=2)
    arrow(xs, -3, xs, -4.5, sa=2, sb=2)
    arrow(xs, -9, xs, -11, enz='methylmalonyl-\nCoA mutase', side=-1, sa=2, sb=2)
    arrow(xs, -14, xs, -15.5, sa=2, sb=2)

    # ================================ footer ================================
    ax.text(X0, -21.5,
            f'Predicted metagenome (PICRUSt2, MetaCyc), n = {m.n}.  Each coloured region is one MetaCyc pathway: solid when it reaches $q$ < 0.05 for the pre-specified score in both the primary and the\n'
            f'coverage-adjusted model (β from the primary model, $q$* from the coverage-adjusted model), dashed when it does so only for one pre-specified genus.  Chips name every pre-specified genus\n'
            f'whose own model survives, with the sign of that genus\' coefficient; the score is NK4A136 + R-7 − $Enterocloster$, so a pathway that falls with the score rises with $Enterocloster$.\n'
            f'Grey nodes are shared intermediates; glycolysis carries no statistic.  Enzymes in italics: PFOR, pyruvate:ferredoxin oxidoreductase; PFL, pyruvate formate-lyase; PTA, phosphotransacetylase;\n'
            f'GDH, glutamate dehydrogenase; host enzymes (FAR1, GNPAT, AGPS) as in Figure 3A.',
            ha='left', va='top', fontsize=fs - 0.7, color=MUTED, linespacing=1.5)
    cx = X0
    for g, (col, _) in GENUS.items():
        t = ax.text(cx, -30.5, g, ha='left', va='center', fontsize=fs - 0.6, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2,rounding_size=0.4', fc=col, ec='none'))
        cx += m._w(t) + 1.8
    ax.text(cx + 0.5, -30.5, 'the three pre-specified genera', ha='left', va='center', fontsize=fs - 0.6, color=MUTED)
    return m
