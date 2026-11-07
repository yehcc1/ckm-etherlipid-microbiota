# -*- coding: utf-8 -*-
"""Figure S5 加上第五個 panel：陰性臂（原 Figure 3D 右半）      20261010_v64

整合圖要接 Figure 3 的位置，原 Figure 3D 的右半——「反方向跑的那一臂」——
不能就這樣消失。S5 的 panel C 只畫最負的 9 條與最正的 9 條，六條陰性臂裡
P162-PWY、PWY-5022、HISDEG-PWY 已在其中，但 PWY-7254、P108-PWY、PWY-4984
不在，所以不能靠 C 代替，要有自己的 panel。

做法：把 XE_figure_metacyc_20260911_v55.py 的原始碼讀進來改，A–D 四個 panel
的程式碼一字不動（避免重畫造成任何差異），只做四件事：
  1. gridspec 從 3 列變 4 列，畫布加高
  2. 在第 4 列放新的 panel E，程式碼照 X5b_figure3_20260912_v59.py 的陰性臂搬過來
  3. 輸出檔名改為 FigureS5_20261010_v64.svg（輸入檔仍是 v55 的結果檔）
  4. 同時輸出 PNG 供目視檢查

陰性臂的六條路徑與 X5b 用同一組斷言把關：q < 0.05、q_nsti_adj < 0.05、beta < 0。
"""
from __future__ import annotations
import sys

SRC = 'XE_figure_metacyc_20260911_v55.py'
OUT = 'XE2_figureS5_v65_generated.py'
s = open(SRC, encoding='utf-8').read()

s_ = open(SRC, encoding='utf-8').read()
PATCHES = [
    # 1. 畫布加高
    ("fig = plt.figure(figsize=(7.2, 9.4))",
     "fig = plt.figure(figsize=(7.2, 9.0))"),
    # 2. gridspec 加一列
    ("gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.45, 1.95], hspace=0.40, wspace=0.34,\n"
     "                      left=0.10, right=0.98, top=0.955, bottom=0.055)",
     "gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.45, 1.30], hspace=0.42,\n"
     "                      wspace=0.34, left=0.10, right=0.98, top=0.955, bottom=0.055)"),
    # 2b. panel C 的標題改掉「survive NSTI adjustment」這個會被誤讀的說法
    ("f'What survives: the {len(M[(M.exposure==\"Microbiota panel\")&(M.q<0.05)&(M.q_nsti_adj<0.05)])} '\n"
     "            f'pathways that remain after NSTI adjustment (9 most negative and 9 most positive shown)'",
     "f'The {len(M[(M.exposure==\"Microbiota panel\")&(M.q<0.05)&(M.q_nsti_adj<0.05)])} pathways '\n"
     "            f'significant in both models (9 most negative and 9 most positive shown)'"),
    # 3. panel A 改成三根柱：base、校正模型自己、兩者交集
    ('''before = [int((M[(M.exposure == e) & (M.q < 0.05)]).shape[0]) for e in EXP]
after = [int((M[(M.exposure == e) & (M.q < 0.05) & (M.q_nsti_adj < 0.05)]).shape[0]) for e in EXP]
x = np.arange(len(EXP))
a.bar(x - 0.19, before, 0.36, color=ORANGE, label='as reported')
a.bar(x + 0.19, after, 0.36, color=BLUE, label='adjusted for NSTI')
for xi, (b_, af) in enumerate(zip(before, after)):
    a.text(xi - 0.19, b_ + 5, str(b_), ha='center', va='bottom', fontsize=6.2, color=INK)
    a.text(xi + 0.19, af + 5, str(af), ha='center', va='bottom', fontsize=6.2, color=INK)''',
     '''# 20261030_v65（外審 issue 04）：原本只畫兩根柱，而且把「兩個模型都顯著」的交集
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
        a.text(xi + dx, v + 5, str(v), ha='center', va='bottom', fontsize=5.8, color=INK)'''),
    # 20261030_v65：Gemmiger 不再當一個 exposure 畫在 A；† 標記也拿掉（不逐條對照人家的圖）
    ("EXP = ['Microbiota panel', 'Gemmiger', 'Ether-lipid ratio', 'CKM stage 3 or 4']",
     "EXP = ['Microbiota panel', 'Ether-lipid ratio', 'CKM stage 3 or 4']"),
    ("a.set_xticks(x); a.set_xticklabels(['Microbiota\\npanel', '$Gemmiger$', 'Ether-lipid\\nratio',",
     "a.set_xticks(x); a.set_xticklabels(['Microbiota\\npanel', 'Ether-lipid\\nratio',"),
    # 4. 輸出檔名
    # 20261030_v65 依裁示：Gemmiger 定向複製整段撤除，他們的是他們的，我們的是自己的發現
    (s_[s_.index('# ---------------- D ----------------'):s_.index("plt.savefig(f'FigureS5_{ST}.svg'")], ''),
    ("plt.savefig(f'FigureS5_{ST}.svg', bbox_inches='tight', pad_inches=0.10); plt.close()\n"
     "print('Figure done')",
     "PANEL_E\n"
     "plt.savefig('FigureS5_20261030_v65.svg', bbox_inches='tight', pad_inches=0.10)\n"
     "plt.savefig('chk/S5_v65.png', dpi=170, bbox_inches='tight', pad_inches=0.10)\n"
     "plt.close()\n"
     "print('Figure S5: A three bars, B, C, D negative arm (Gemmiger replication removed)')"),
]

PANEL_E = '''
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
       'The carbohydrate-fermentation arm in panel C runs one way and these six run the\\n'
       'other, so the panel is not a single gradient of "more bacteria, more pathways".\\n'
       f'Of the {n_neg_all + n_pos_all} pathways surviving adjustment for reference-genome coverage, '
       f'{n_pos_all} rise\\nwith the microbiota panel and {n_neg_all} fall; the six drawn here are the '
       'named\\nprotein- and organic-acid-degradation routes that fall, three of which\\n'
       '(P162-PWY, PWY-5022, HISDEG-PWY) also appear among the nine most negative\\n'
       'pathways in panel C while the other three do not.\\n\\n'
       'β is from the primary model and $q$* from the model additionally adjusted for\\n'
       f'reference-genome coverage; every pathway shown reaches $q$ < 0.05 in both.\\n'
       + dag_note +
       f'Predicted metagenome (PICRUSt2), n = {int(P.n.iloc[0]):,}.',
       transform=g.transAxes, fontsize=6.2, color=MUTED, va='top', ha='left', linespacing=1.75)
'''

for old, new in PATCHES:
    if s.count(old) != 1:
        sys.exit(f'!! 片段出現 {s.count(old)} 次（預期 1）：{old[:60]}…')
    s = s.replace(old, new)
s = s.replace('PANEL_E', PANEL_E)
open(OUT, 'w', encoding='utf-8').write(s)
print(f'寫出 {OUT}')
