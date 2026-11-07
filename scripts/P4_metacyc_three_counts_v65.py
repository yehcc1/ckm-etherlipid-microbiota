# -*- coding: utf-8 -*-
"""MetaCyc 的三種數量分開記錄，陰性結論改成依模型規格而定      20261030_v65

外審 issue 04 與 05。

issue 04：原本只存兩個數字——base 模型的 q<0.05 數，以及「兩個模型都 q<0.05」
的交集數，但後者在圖上被標成 adjusted for NSTI，讀起來像是校正模型自己的發現數。
三個數字其實都該有：

  微生物面板   base 251   校正模型自己 115   交集 69
  Gemmiger     base  96   校正模型自己  38   交集 32
  醚脂比值     base   7    校正模型自己   1   交集  1
  CKM 3 或 4   base   0    校正模型自己   0   交集  0

issue 05：原稿寫成「所有 pathway–metabolite 分析皆為陰性」。實際上 base 模型
確實一條都沒有，但 NSTI 校正後 S20 有 6 條、S21 有 5 條達 q<0.05，而且幾乎
全是 arginine 相關。為零的是「兩個模型都顯著」的交集。要求兩者都顯著是事前
定下的保守規則，這點可以守；但不能把校正後才出現的訊號講成不存在。

值得注意的是那幾條的 base q 落在 0.06 到 0.30，是校正之後才變顯著的，所以只能
當探索性線索。這支腳本把數字算出來並另存一張清單，不改任何模型。
"""
from __future__ import annotations
import sys
import pandas as pd

SRC = 'XD_metacyc_enrichment_20260911_v55.py'
OLD = """for nm in EXP:
    add(f'metacyc_{nm.split()[0].lower()}_survive_nsti',
        int(((M.exposure == nm) & (M.q < 0.05) & (M.q_nsti_adj < 0.05)).sum()), 'pathways', '')
"""
NEW = """for nm in EXP:
    sel = M.exposure == nm
    # 20261030_v65：三種數量分開存，圖與正文才不會把交集講成校正模型的發現數
    add(f'metacyc_{nm.split()[0].lower()}_n_q05_base',
        int((sel & (M.q < 0.05)).sum()), 'pathways', 'q<0.05 in the primary model')
    add(f'metacyc_{nm.split()[0].lower()}_n_q05_adjmodel',
        int((sel & (M.q_nsti_adj < 0.05)).sum()), 'pathways',
        'q<0.05 in the NSTI-adjusted model, irrespective of the primary model')
    add(f'metacyc_{nm.split()[0].lower()}_survive_nsti',
        int((sel & (M.q < 0.05) & (M.q_nsti_adj < 0.05)).sum()), 'pathways',
        'q<0.05 in BOTH models (the intersection)')
"""

s = open(SRC, encoding='utf-8').read()
if s.count(OLD) != 1:
    sys.exit(f'!! XD 的 survive_nsti 片段找到 {s.count(OLD)} 次（預期 1）')
open(SRC, 'w', encoding='utf-8').write(s.replace(OLD, NEW))
print('XD 已改：每個 exposure 存 base／adjusted 模型／交集三個數量')

# ---------------- 校正後才出現的 pathway–metabolite 訊號清單 ----------------
rows = []
for f, src in [('TableS20_pathway_metabolite_pairs_20260911_v56.csv', 'Matched pathway–metabolite pairs'),
               ('TableS21_pathway_metabolite_broad_20260911_v56.csv', 'Broad pathway–metabolite scan')]:
    d = pd.read_csv(f)
    n_base = int((d.q < 0.05).sum())
    n_adj = int((d.q_nsti_adj < 0.05).sum())
    n_both = int(((d.q < 0.05) & (d.q_nsti_adj < 0.05)).sum())
    print(f'{src}: n={len(d)}  base {n_base}  adjusted模型 {n_adj}  交集 {n_both}')
    sub = d[d.q_nsti_adj < 0.05].copy()
    sub.insert(0, 'table', src)
    rows.append(sub)
OUT = pd.concat(rows, ignore_index=True)
cols = [c for c in ['table', 'pathway', 'name', 'metabolite', 'beta', 'se', 'p', 'q',
                    'p_nsti_adj', 'q_nsti_adj', 'n'] if c in OUT.columns]
OUT = OUT[cols].sort_values('q_nsti_adj')
OUT.to_csv('TableSz_adjusted_only_pairs_20261030_v65.csv', index=False)
assert (OUT.q >= 0.05).all(), '這張表只收 base 模型未達標、校正後才達標的配對'
print(f'\n寫出 TableSz_adjusted_only_pairs_20261030_v65.csv（{len(OUT)} 列，'
      f'base q 範圍 {OUT.q.min():.3f}–{OUT.q.max():.3f}）')
print(OUT.to_string(index=False))
