# -*- coding: utf-8 -*-
"""Figure 3（整合圖）v65 改版                                      20261030_v65

在 out/Figure3_20261030_v65.svg 上動三處，不重畫既有內容：
1. panel c：R1 那兩行改以事前三菌屬分數為主、CV 面板為次（數字讀自 TableS15 / numbers 檔）。
2. panel e：拿掉 CKM 0–4 光譜圖，改畫「鏈的終點在哪一刀」——
   左：R1 → 四個分期切點；右：事前分數 → 四個分期切點（直接效果），
   並註明 ≥2 那一刀經八比值中介的比例（TableSh）。資料：TableSq、TableSh，n = 743。
3. 註腳：刪掉「CKM illustrations show representative phenotypes」一句。
4.（20261031 第 k 輪）整張圖的暴露一律改成事前三菌屬分數——v64 原稿 panel a／c／d 的數字
   都是 CV 面板的，與現行正文的主要暴露不一致。路徑統計改讀 TableSmc（X28，β 主模型、
   q* 覆蓋度校正模型，未存活者改列扛它的事前菌屬），類別統計改讀 TableS4 的事前分數列，
   panel d 的散點與聯合模型改用事前分數重畫。另：移除基因型標籤（論文已不談基因型）。
所有數字讀自結果檔。
"""
from __future__ import annotations
import re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ST = '20261030_v65'
SRC = 'Figure3_20261010_v64.svg'   # v64 整合圖原稿（五 panel、尚未動刀），手術每次從這裡重做
OUT = 'GraphicalAbstract_20261107_v66.svg'   # 20261031：整合圖改作 graphical abstract
GREEN, INK, MUTED, BLUEISH, PURPLE = '#287F72', '#243F4B', '#647985', '#3D7DAD', '#8478A5'
N = pd.concat([pd.read_csv(f'numbers_v65_all_{ST}.csv'), pd.read_csv(f'numbers_extra_{ST}.csv'),
               pd.read_csv('numbers_metacyc_taxa_20261031_v65.csv')]).drop_duplicates('key', keep='last').set_index('key').value
nv = lambda k: float(N[k])
S15 = pd.read_csv(f'out/TableS15_pc_sensitivity_{ST}.csv') if False else None


def sup(v: float) -> str:
    m, e = f'{v:.1e}'.split('e')
    t = str(int(e)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return f'{m} × 10{t}'


def pfmt(p: float) -> str:
    return f'P = {p:.3f}' if p >= 0.001 else f'P = {sup(p)}'


def sgn(b: float, d=3) -> str:
    return f'{b:+.{d}f}'.replace('-', '−')


s = open(SRC, encoding='utf-8').read()


def swap(old_id: str, new_text: str):
    global s
    pat = re.compile(rf'(<text id="{old_id}"[^>]*>\s*<tspan[^>]*>)(.*?)(</tspan>)', re.S)
    m = pat.search(s)
    if not m: sys.exit(f'!! 找不到 {old_id}')
    print(f'  {old_id}: {m.group(2).strip()}  →  {new_text}')
    s = s[:m.start(2)] + new_text + s[m.end(2):]


# ------------------------------------------------------- 1. panel c：R1 兩行
pre_b, pre_p = nv('rob_pre_base_beta'), nv('rob_pre_base_p')
cv_b, cv_p = nv('rob_cv_base_beta'), nv('rob_cv_base_p')
S15 = pd.read_csv(f'out/TableS15_pc_sensitivity_{ST}.csv')
g_pre = S15[S15.result.str.startswith('Pre-specified score → ether')].iloc[0]
g_cv = S15[S15.result.str.startswith('Microbiota panel → ether')].iloc[0]
assert abs(g_pre.beta_genotyped_with_pcs - nv('geno_pre_beta_pcs')) < 1e-9
swap('panel-r1-association',
     f'Pre-specified three-genus score: β {sgn(pre_b)}; {pfmt(pre_p)}  ·  CV panel: β {sgn(cv_b)}; {pfmt(cv_p)}')
# 20261031_v65：遺傳校正那一行整個拿掉（祖源校正已併進 Table S6 穩健度階梯，論文不再談基因型）
_m = re.search(r'\s*<text id="genotype-sensitivity".*?</text>', s, flags=re.S); assert _m; s = s[:_m.start()] + s[_m.end():]
# panel-r1-association 原本是 19pt、藍色；兩段併一行後縮一點
s = s.replace('<text id="panel-r1-association" x="1734" y="1256" font-size="19"',
              '<text id="panel-r1-association" x="1734" y="1256" font-size="18"', 1)

# ------------------------------------------------------- 1b. panel a／c／d：暴露一律改成事前分數
MC = pd.read_csv('TableSmc_metacyc_taxa_20261031_v65.csv')
MCP = MC[MC.exposure == 'Pre-specified score'].set_index('pathway')
MCG = {'NK4A136': MC[MC.exposure == 'Lachnospiraceae NK4A136 group'].set_index('pathway'),
       'R-7': MC[MC.exposure == 'Christensenellaceae R-7 group'].set_index('pathway')}
CLS = pd.read_csv(f'out/TableS4_class_specificity_{ST}.csv')
SPC = pd.read_csv(f'out/TableS5_species_{ST}.csv')


def qsup(q: float) -> str:
    return f'{q:.3g}' if q >= 0.01 else sup(q)


def pstat(pw: str) -> str:
    """與 Figure 4 同一條規則：分數兩個模型都 q<0.05 就報分數，否則報扛它的事前菌屬。"""
    r = MCP.loc[pw]
    if r.survives:
        return f'β {sgn(r.beta, 2)}; q* {qsup(r.q_nsti_adj)}'
    car = [g for g in MCG if MCG[g].loc[pw, 'survives'] and np.sign(MCG[g].loc[pw, 'beta']) == np.sign(r.beta)]
    assert car, f'{pw}: 分數與兩個正向菌屬都沒存活'
    rg = MCG[car[0]].loc[pw]
    return f'{car[0]} {sgn(rg.beta, 2)}; q* {qsup(rg.q_nsti_adj)}'


def cls_stat(phen: str) -> str:
    r = CLS[(CLS.phenotype == phen) & (CLS.score == 'Pre-specified score')].iloc[0]
    return f'β {sgn(r.beta, 3)}; {pfmt(r.p)}'


swap('microbiome-sample', f'Pre-specified three-genus score · n = {int(nv("mct_n")):,}')
swap('microbiome-model', 'β: primary model; q*: coverage-adjusted model')
swap('legend-statistics',
     'Microbial β values are from the primary model and q* values from models additionally adjusted for '
     'reference-genome coverage; where a genus is named, the pathway is carried by that genus rather than by the '
     'score itself. PC ae does not resolve plasmanyl versus plasmenyl species.')
for _id, _pw in [('sucrose-IV-stat', 'PWY-5384'), ('sucrose-III-stat', 'PWY-621'),
                 ('bifidobacterium-shunt-stat', 'P124-PWY'), ('mixed-acid-stat', 'FERMENTATION-PWY'),
                 ('heterolactic-stat', 'P122-PWY'), ('acetone-stat', 'PWY-6588'),
                 ('palmitate-stat', 'PWY-5971')]:
    swap(_id, pstat(_pw))
swap('pwy-r1-unique', f'PWY-5971: the only R1-associated pathway among {int(nv("n_metacyc_pathways"))} tested')
swap('pwy-r1-nsti', f'after adjustment for reference-genome coverage (q* = {nv("pwy5971_r1_q_nsti"):.3g})')
swap('pwy-r1-phenotype-model',
     f'PWY-5971–R1 phenotype model: β {sgn(nv("pwy5971_ether-lipid_ratio_beta"))}; '
     f'{pfmt(nv("pwy5971_ether-lipid_ratio_p"))}; n = {int(nv("n_microbiome")):,}')
swap('plasma-total-pcae-stat', cls_stat('Total PC ae'))
swap('plasma-total-pcaa-stat', cls_stat('Total PC aa'))
swap('class-ratio-stat', 'PC ae / PC aa: ' + cls_stat('PC ae / PC aa'))
swap('class-exposure-note',
     f'Pre-specified score · predicted pathway versus total PC aa: {pfmt(nv("pwy5971_total_pc_aa_p"))}')
swap('nuclear-dna-label', 'Host cell nucleus')
swap('legend-limitations',
     'Microbial precursor transfer, host synthesis flux and temporal ordering were not established.'
     ' CKM illustrations show representative phenotypes.')

# panel d：聯合模型兩行與散點都改成事前分數
_old_joint = ('Joint model on the 37 species-level effects: carbon number −0.0041 per carbon, P = 0.0176')
assert s.count(_old_joint) == 1
s = s.replace(_old_joint, f'Joint model on the {int(nv("spc_pre_n"))} species-level effects: carbon number '
              f'{sgn(nv("carbon_pre_beta"), 4)} per carbon, P = {nv("carbon_pre_p"):.3f}')
_old_db = 'double bonds P = 0.21; n = 743'
assert s.count(_old_db) == 1
s = s.replace(_old_db, f'double bonds P = {nv("doublebond_pre_p"):.2f}; n = {int(nv("n_microbiome")):,}')

_d = SPC[(SPC.group == 'PC ae / total lysoPC') & (SPC.score == 'Pre-specified score')].copy()
assert len(_d) == int(nv('spc_pre_n'))
_cn = _d.species.str.extract(r'C(\d+):(\d+)').astype(float)
_d['carbon'] = _cn[0].values
_q05 = int((_d.q < 0.05).sum()); assert _q05 == int(nv('spc_pre_q05'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'font.size': 15})
_fig = plt.figure(figsize=(980 / 72, 300 / 72))
_ax = _fig.add_axes([0.10, 0.17, 0.87, 0.78])
_ax.axhline(0, color='#C6D5DE', lw=1.2, ls=(0, (4, 3)))
_ax.scatter(_d.carbon + np.random.default_rng(2).normal(0, 0.12, len(_d)), _d.beta, s=52,
            c=[GREEN if q < 0.05 else 'white' for q in _d.q], edgecolors=GREEN, linewidths=1.4, zorder=3)
_z = np.polyfit(_d.carbon, _d.beta, 1); _xx = np.linspace(_d.carbon.min(), _d.carbon.max(), 20)
_ax.plot(_xx, np.polyval(_z, _xx), color='#B4503C', lw=1.8, zorder=2)
_ax.set_xlabel('PC ae carbon number', color=INK, fontsize=14)
_ax.set_ylabel('Effect on species / total lysoPC\n(SD per SD)', color=INK, fontsize=13)
_ax.tick_params(colors=MUTED, labelsize=13)
for _sp in ('top', 'right'): _ax.spines[_sp].set_visible(False)
for _sp in ('left', 'bottom'): _ax.spines[_sp].set_color('#C6D5DE')
_h = [plt.Line2D([], [], marker='o', ls='', mfc=GREEN, mec=GREEN, ms=7,
                 label=f'q < 0.05 ({_q05} of {len(_d)})'),
      plt.Line2D([], [], marker='o', ls='', mfc='white', mec=GREEN, ms=7,
                 label=f'q ≥ 0.05 ({len(_d) - _q05})')]
_ax.legend(handles=_h, frameon=False, fontsize=12, loc='upper right', labelcolor=MUTED)
_fig.savefig('chk/_panel_d.svg', transparent=True)
plt.close(_fig)
_inner = open('chk/_panel_d.svg', encoding='utf-8').read()
_inner = _inner[_inner.index('>', _inner.index('<svg')) + 1: _inner.rindex('</svg>')]
_inner = re.sub(r'<!--.*?-->', '', _inner, flags=re.S)
_inner = re.sub(r'<metadata>.*?</metadata>', '', _inner, flags=re.S)
_i0 = s.index('<svg x="100" y="1390" width="980" height="300"')
_depth, _j = 0, _i0
while True:
    _m = re.compile(r'<svg\b|</svg>').search(s, _j)
    assert _m, 'panel d 的內嵌 svg 找不到結尾'
    _depth += 1 if _m.group(0) == '<svg' else -1
    _j = _m.end()
    if _depth == 0: break
s = (s[:_i0] + f'<svg x="100" y="1390" width="980" height="300" viewBox="0 0 980 300" '
     f'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
     f'overflow="visible">{_inner}</svg>' + s[_j:])

# ------------------------------------------------------- 2. panel e
swap('panel-label-e', 'Where the chain reaches CKM staging')
swap('ckm-association',
     f'Higher R1 associated with lower CKM stage (0–4): β {sgn(nv("r2o_R1_CKMstage5level_beta"))}; '
     f'q = {sup(nv("r2o_R1_CKMstage5level_q"))}; n = {int(nv("r2o_R1_CKMstage5level_n")):,}')

import glob
SQ = pd.read_csv(sorted(glob.glob(f'TableSq_*_{ST}.csv'))[0])
cuts = ['stage >= 1', 'stage >= 2', 'stage >= 3', 'stage >= 4']
lab = ['≥ 1', '≥ 2', '≥ 3', '≥ 4']
W, H = 2200, 430
EXTRA = 240                       # 畫布加高、註腳下移
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'axes.linewidth': 1.0, 'font.size': 15})
fig = plt.figure(figsize=(W / 72, H / 72))
axs = [fig.add_axes([0.07, 0.20, 0.36, 0.62]), fig.add_axes([0.59, 0.20, 0.36, 0.62])]
for ax, ex, col, title in [(axs[0], 'Ether-lipid ratio', BLUEISH, 'R1 → CKM stage boundary'),
                           (axs[1], 'Pre-specified score', GREEN, 'Microbiota score → CKM stage boundary (direct)')]:
    d = SQ[SQ.exposure == ex].set_index('boundary').loc[cuts]
    x = np.arange(4)
    ax.axhline(0, color='#C6D5DE', lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.errorbar(x, d.beta, yerr=1.96 * d.se, fmt='none', ecolor=col, lw=1.8, capsize=5, zorder=2)
    ax.scatter(x, d.beta, s=90, c=[col if p < 0.05 else 'white' for p in d.p], edgecolors=col, linewidths=1.8, zorder=3)
    for xi, (_, r) in zip(x, d.iterrows()):
        ax.text(xi, r.beta - 1.96 * r.se - 0.08, pfmt(r.p).replace('P = ', 'P = '), ha='center', va='top',
                fontsize=13, color=INK if r.p < 0.05 else MUTED)
    ax.set_xticks(x); ax.set_xticklabels([f'CKM {l}  (events {int(e)})' for l, e in zip(lab, d.n_events)], fontsize=14)
    ax.set_ylabel('log-odds per SD\n(95% CI)', color=INK, fontsize=14)
    ax.set_title(f'{title}   ·   n = {int(d.n.iloc[0])}', color=INK, fontsize=17, fontweight='bold', loc='left', pad=8)
    ax.tick_params(colors=MUTED, labelsize=13)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color('#C6D5DE')
    ax.set_xlim(-0.5, 3.5)
axs[0].set_ylim(-1.25, 0.95); axs[1].set_ylim(-1.25, 0.95)
# 右圖註：≥2 那一刀的中介
ind, lo, hi, prop = (nv('pm_pre_ge2_total_indirect'), nv('pm_pre_ge2_ind_lo'), nv('pm_pre_ge2_ind_hi'),
                     nv('pm_pre_ge2_prop_mediated'))
axs[1].text(1, 0.78, f'indirect via the eight ratios: {sgn(ind)} ({sgn(lo)} to {sgn(hi)})',   # 20261107_v66：不印 % mediated
            ha='center', va='center', fontsize=13, color=PURPLE,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=PURPLE, lw=0.9))
axs[1].annotate('', xy=(1, 0.12), xytext=(1, 0.62), arrowprops=dict(arrowstyle='-|>', color=PURPLE, lw=1.2))
fig.savefig('chk/_panel_e.svg', transparent=True)
plt.close(fig)
inner = open('chk/_panel_e.svg', encoding='utf-8').read()
inner = inner[inner.index('>', inner.index('<svg')) + 1: inner.rindex('</svg>')]
inner = re.sub(r'<!--.*?-->', '', inner, flags=re.S)
inner = re.sub(r'<metadata>.*?</metadata>', '', inner, flags=re.S)

i0 = s.index('<g id="ckm-stage-0"')
i1 = s.index('<path d="M 48 1650 H 2350"')
mid = (f'<svg x="100" y="1385" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" overflow="visible">{inner}</svg>\n  '
       f'<text x="1200" y="{1385 + H + 22}" font-size="16" font-weight="400" text-anchor="middle" fill="{MUTED}" '
       f'font-family="Arial, Helvetica, sans-serif">Filled: P &lt; 0.05. Models adjusted for age, sex, sequencing batch and BMI. '
       f'The microbiota score has no detectable direct effect on any boundary; its effect on the stage-2 boundary runs through the eight ratios.</text>\n  ')
tail = s[i1:]
j_end = tail.index('</g>')                 # 原 translate(0,400) 群組的結尾
tail = f'<g transform="translate(0,{EXTRA})">\n' + tail[:j_end] + '</g>\n' + tail[j_end:]
s = s[:i0] + mid + tail
m = re.search(r'viewBox="0 0 2400 (\d+)"', s); h0 = int(m.group(1))
s = s.replace(m.group(0), f'viewBox="0 0 2400 {h0 + EXTRA}"', 1)
m = re.search(r'height="([\d.]+)mm"', s)
s = s.replace(m.group(0), f'height="{float(m.group(1)) * (h0 + EXTRA) / h0:.1f}mm"', 1)
# 刪掉 defs 裡沒人用的 stage 圖樣？保留，無害。

# ------------------------------------------------------- 3. 註腳：刪掉示意圖那一句（legend-limitations 已重寫）
old = ' CKM illustrations show representative phenotypes.'
assert s.count(old) == 1, s.count(old)
s = s.replace(old, '')

# 原稿 metadata 裡留著 v64 製圖時的修訂紀錄（含已撤除分析的字樣），交付檔不帶出去
_md = re.search(r'\s*<metadata\b[^>]*>.*?</metadata>', s, flags=re.S)
assert _md, '找不到原稿的 metadata 區塊'
if _md: s = s[:_md.start()] + s[_md.end():]
for _w in ('Gemmiger', 'genotype', 'progression'):
    assert _w not in s, f'graphical abstract 仍殘留「{_w}」'

open(OUT, 'w', encoding='utf-8').write(s)
print(f'寫出 {OUT}')
