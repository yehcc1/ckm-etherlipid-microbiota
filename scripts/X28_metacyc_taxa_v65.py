# -*- coding: utf-8 -*-
"""MetaCyc 路徑富集：事前三菌屬分數與三個事前菌屬各自（Wu 2025 的作法：鑑定出的菌 → 它帶什麼路徑） 20261031_v65

沿用 XD 的全部前處理（PICRUSt2 路徑表 CLR、NSTI、共變數、hc3），只把暴露換成：
  Pre-specified score（主要暴露）、NK4A136、Christensenellaceae R-7、Enterocloster（CLR 標準化）、
  CV panel、R1（與 XD 相同，作為核對）。
每一條路徑：主要模型與加權 NSTI 校正模型各一次；只有兩者都 q < 0.05 才算存活（與 XD 相同的規則）。
路徑分類以 MetaCyc 名稱的關鍵字歸類（amino acid / carbohydrate & fermentation / nucleotide / lipid & fatty acid /
cofactor & vitamin / cell wall & envelope / energy / other），只用於作圖分組。
輸出：TableSmc_metacyc_taxa_{ST}.csv、numbers_metacyc_taxa_{ST}.csv
"""
from __future__ import annotations
import re, gzip, numpy as np, pandas as pd
src = open('XD_metacyc_enrichment_20260911_v55.py', encoding='utf-8').read()
head = src[:src.index("EXP = {'Microbiota panel': panel")]
head = head.replace("SC = pd.read_csv('panel_scores_20260911_v54.csv')", "SC = pd.read_csv('out/panel_scores_20261030_v65.csv')")
g = {'__name__': 'xd_head'}; exec(compile(head, 'XD_head', 'exec'), g)
scan, GZ, S, panel, r1, nz_src, nsti, rint = g['scan'], g['GZ'], g['S'], g['panel'], g['r1'], None, g['nsti'], g['rint']
SC = g['SC']; bh = g['bh']
ST = '20261031_v65'
prior = SC.score_prior.reindex(S)
P3 = ['NK4A136', 'Christensenellaceae_R-7', 'Enterocloster']
def findg(tok):
    hit = [c for c in GZ.columns if tok.lower().replace('_', ' ') in c.lower().replace('_', ' ')]
    assert hit, tok; return hit[0]
genera = {findg(t): GZ[findg(t)].reindex(S) for t in P3}
print('三個事前菌屬：', list(genera))
EXP = {'Pre-specified score': prior, **genera, 'CV panel': panel, 'Ether-lipid ratio': r1}
nz = rint(nsti.values)
rows = []
for nm, x in EXP.items():
    D = scan(x); D2 = scan(x, extra=nz)
    D = D.merge(D2[['pathway', 'beta', 'p', 'q']], on='pathway', suffixes=('', '_nsti_adj'))
    D.insert(0, 'exposure', nm); rows.append(D)
M = pd.concat(rows, ignore_index=True)
NAME = dict(l.split('\t', 1) for l in gzip.open('metacyc_names_from_S17.txt.gz', 'rt', encoding='utf-8', errors='replace').read().rstrip('\n').split('\n'))
M.insert(2, 'name', M.pathway.map(NAME))
# 核對：CV panel 與 R1 的結果要和 Table S17 一致
S17 = pd.read_csv('TableS17_metacyc_enrichment_20260911_v55.csv')
for nm, nm17 in [('CV panel', 'Microbiota panel'), ('Ether-lipid ratio', 'Ether-lipid ratio')]:
    a = M[M.exposure == nm].set_index('pathway'); b = S17[S17.exposure == nm17].set_index('pathway')
    assert np.allclose(a.beta.reindex(b.index), b.beta, atol=1e-9) and np.allclose(a.q_nsti_adj.reindex(b.index), b.q_nsti_adj, atol=1e-9), nm
M['survives'] = (M.q < 0.05) & (M.q_nsti_adj < 0.05)


def category(name: str) -> str:
    n = name.lower()
    if re.search(r'amino acid|l-[a-z]+ (bio|deg)|arginine|lysine|histidine|isoleucine|leucine|valine|methionine|threonine|tryptophan|tyrosine|phenylalanine|glutamate|glutamine|aspartate|asparagine|serine|glycine|cysteine|proline|alanine|ornithine|polyamine|putrescine|4-aminobutanoate|urea cycle|chorismate', n): return 'Amino acid metabolism'
    if re.search(r'sucrose|starch|glycogen|galact|lactose|glucose|fructose|mannose|xylose|rhamnose|fucose|glycolysis|fermentation|pyruvate|acetyl|acetate|lactate|propanoate|butanoate|butyrate|glyoxylate|tca|entner|pentose|gluconeogenesis|hexitol|sugar|trehalose|chitin|cellulose|pectin|mannan|xylan|glucuronate|galacturonate|citrate|succinate|methylglyoxal|acetone|acetylene|ethanol|methanol|formaldehyde|glycerol', n): return 'Carbohydrate metabolism and fermentation'
    if re.search(r'purine|pyrimidine|nucleoside|nucleotide|adenosine|guanosine|inosine|dtdp|udp-|cmp-|gdp-|ribonucle|deoxyribonucle|thymidine|uridine|cytidine', n): return 'Nucleotide metabolism'
    if re.search(r'fatty acid|palmitate|oleate|stearate|lipid|phospholipid|mycolate|cis-vaccenate|biotin|lipoate|lipid a|kdo|isoprene|undecaprenyl|polyprenyl|farnesyl|geranylgeranyl|methylerythritol|mevalonate|sterol', n): return 'Lipid, fatty acid and isoprenoid metabolism'
    if re.search(r'cofactor|vitamin|folate|thiamin|riboflavin|flavin|nad|nicotinate|pantothenate|coenzyme|cobalamin|adenosylcobalamin|heme|siroheme|menaquinone|ubiquinone|phylloquinone|pyridoxal|tetrahydrofolate|molybdenum|thiazole|s-adenosyl|pyridox|queuosine|tetrapyrrole|porphyrin', n): return 'Cofactor and vitamin biosynthesis'
    if re.search(r'peptidoglycan|cell wall|teichoic|lipopolysaccharide|o-antigen|enterobacterial|colanic|capsul|exopolysaccharide|murein|glycan|heptose|mycolyl', n): return 'Cell envelope biosynthesis'
    if re.search(r'respiration|electron|oxidative phosphorylation|photosynth|methanogen|nitrate|nitrogen|sulfate|sulfur|hydrogen|carbon monoxide|reductive|calvin|fixation|denitrification', n): return 'Energy metabolism'
    return 'Other'


M['category'] = M.name.map(category)
M.to_csv(f'TableSmc_metacyc_taxa_{ST}.csv', index=False)
num = []
pd.set_option('display.width', 220)
for nm in EXP:
    d = M[M.exposure == nm]; s = d[d.survives]
    num += [dict(key=f'mct_{nm}_n_q05', value=int((d.q < 0.05).sum()), source='X28'),
            dict(key=f'mct_{nm}_n_survive', value=int(s.shape[0]), source='X28'),
            dict(key=f'mct_{nm}_n_survive_pos', value=int((s.beta > 0).sum()), source='X28'),
            dict(key=f'mct_{nm}_n_survive_neg', value=int((s.beta < 0).sum()), source='X28')]
    print(f'\n== {nm}: q<0.05 主要模型 {int((d.q < 0.05).sum())}；兩者都過 {s.shape[0]}（正 {int((s.beta > 0).sum())}／負 {int((s.beta < 0).sum())}）')
    print(s.sort_values('p').head(15)[['pathway', 'name', 'category', 'beta', 'q', 'q_nsti_adj']].to_string(index=False))
    if nm == 'Pre-specified score':
        print('\n  分類計數：'); print(s.category.value_counts().to_string())
        r = d[d.pathway == 'PWY-5971'].iloc[0]
        print(f'\n  PWY-5971：β {r.beta:+.3f} q {r.q:.2e} q_nsti {r.q_nsti_adj:.2e}')
        num += [dict(key='mct_pre_pwy5971_beta', value=r.beta, source='X28'), dict(key='mct_pre_pwy5971_q', value=r.q, source='X28'),
                dict(key='mct_pre_pwy5971_q_nsti', value=r.q_nsti_adj, source='X28')]
TAG = {'Pre-specified score': 'pre', 'Lachnospiraceae NK4A136 group': 'nk', 'Christensenellaceae R-7 group': 'r7', 'Enterocloster': 'ec', 'CV panel': 'cv', 'Ether-lipid ratio': 'r1'}
PW = {'PWY-5971': 'pwy5971', 'PWY-5384': 'suc4', 'PWY-621': 'suc3', 'P124-PWY': 'bifido', 'FERMENTATION-PWY': 'mixed', 'P122-PWY': 'hetero',
      'PWY-6588': 'acetone', 'P162-PWY': 'glu', 'PWY-5022': 'gaba', 'HISDEG-PWY': 'his', 'PWY-7254': 'tca7', 'P108-PWY': 'prop'}
for nm, tg in TAG.items():
    d = M[M.exposure == nm].set_index('pathway')
    for pw, pt in PW.items():
        r = d.loc[pw]
        num += [dict(key=f'mct_{tg}_{pt}_beta', value=r.beta, source='X28'), dict(key=f'mct_{tg}_{pt}_q', value=r.q, source='X28'),
                dict(key=f'mct_{tg}_{pt}_qn', value=r.q_nsti_adj, source='X28'), dict(key=f'mct_{tg}_{pt}_surv', value=int(r.survives), source='X28')]
    num += [dict(key=f'mct_{tg}_n_q05', value=int((d.q < 0.05).sum()), source='X28'), dict(key=f'mct_{tg}_n_survive', value=int(d.survives.sum()), source='X28'),
            dict(key=f'mct_{tg}_n_survive_pos', value=int((d.survives & (d.beta > 0)).sum()), source='X28'),
            dict(key=f'mct_{tg}_n_survive_neg', value=int((d.survives & (d.beta < 0)).sum()), source='X28')]
# 分類計數（事前分數）
d = M[(M.exposure == 'Pre-specified score') & M.survives]
for c, k in d.category.value_counts().items():
    num.append(dict(key='mct_pre_cat_' + c.split()[0].lower().rstrip(','), value=int(k), source='X28'))
# 三菌屬之間：存活路徑的重疊
sv = {TAG[g]: set(M[(M.exposure == g) & M.survives].pathway) for g in ['Lachnospiraceae NK4A136 group', 'Christensenellaceae R-7 group', 'Enterocloster']}
pre_sv = set(d.pathway)
num.append(dict(key='mct_pre_carried_by_any_genus', value=int(sum(1 for p in pre_sv if p in sv['nk'] | sv['r7'] | sv['ec'])), source='X28'))
# 事前分數存活路徑中，方向與 Enterocloster 相反的比例
ec = M[M.exposure == 'Enterocloster'].set_index('pathway'); pre = M[M.exposure == 'Pre-specified score'].set_index('pathway')
opp = sum(1 for p in pre_sv if ec.loc[p, 'survives'] and np.sign(ec.loc[p, 'beta']) == -np.sign(pre.loc[p, 'beta']))
num.append(dict(key='mct_pre_ec_opposite', value=int(opp), source='X28')); num.append(dict(key='mct_pre_ec_survive_any', value=int(sum(1 for p in pre_sv if ec.loc[p, 'survives'])), source='X28'))
_n = M[M.exposure != 'Ether-lipid ratio'].n; assert _n.nunique() == 1
num.append(dict(key='mct_n', value=int(_n.iloc[0]), source='X28'))
pd.DataFrame(num).drop_duplicates('key', keep='last').to_csv(f'numbers_metacyc_taxa_{ST}.csv', index=False)
print(f'\n寫出 TableSmc_metacyc_taxa_{ST}.csv')
