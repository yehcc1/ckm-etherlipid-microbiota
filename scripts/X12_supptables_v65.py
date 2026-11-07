# -*- coding: utf-8 -*-
"""把 v65 的補充表打包成一份活頁簿      20261030_v65

每一張表都直接取自本次重跑管線產出的 CSV，不再是事後修補的版本。
S1（比值定義）與 S7（事前登錄）是靜態文件，照原樣帶過來並在 Contents 註明。
S9 在 v52 時是用 v45 那版「基因型為入選條件」的樣本做的，已於 X11 用現行
分析集（MB825／MB743／METAB）與大表 V10 的分期重做。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ST = '20260911_v54'          # 管線內部戳記（本輪重跑後的輸出仍用這個戳記）
OUT_ST = '20261030_v65'
MAN = '/mnt/user-data/uploads/CKM+microbiome/analysis/manuscript'

#           工作表名稱                        來源檔                                      說明
SHEETS = [
    ('Table S1 ratio definitions',  f'{MAN}/TableS1_ratio_definitions_20260910_v45.csv',
     '八個比值的定義（靜態文件，未隨本次重算改變）'),
    ('Table S2 specificity',        f'TableS2_specificity_{ST}.csv',
     '139 菌屬 × 8 個比值表型的全掃描'),
    ('Table S3 specificity q05',    f'TableS3_specificity_q05_{ST}.csv',
     'S2 中 q < 0.05 的子集'),
    ('Table S4 class specificity',  f'TableS4_class_specificity_{ST}.csv',
     '脂質類別層級的特異性'),
    ('Table S5 species',            f'TableS5_species_{ST}.csv',
     '★ BH 指派錯誤已在 X2 原始碼修正（原本 174 列中 74 列 q < p）；'
     '以 (group × score) 為檢定家族，比值家族 CV panel 為 31/37'),
    ('Table S6 robustness',         f'TableS6_robustness_{ST}.csv',
     '用藥、組成梯度、遺傳祖源主成分（585 人）與三酸甘油酯校正後的穩健度；20261031 追加祖源列'),
    ('Table S7 pre-specification',   'TableS7_prereg_20261107_v66.csv',   # 主圖編號已依 v66（Fig 1 與 2 合併）改
     '事前指定項目對照（靜態文件，未隨本次重算改變）'),
    ('Table S8 ratio-outcome',      'out/TableS8_ratio_outcome_20261107_v66.csv',   # 20261107：無分期者不再當成非事件
     '八個比值對臨床結局；CKM 分期改用大表 V10'),
    ('Table S9 subgroup sensitivity', f'TableS9_subgroup_sensitivity_{ST}.csv',
     '★ 已在現行分析集（MB825／MB743／METAB）與大表 V10 分期上重做。'
     'v52 版的 dx_queried==1 切法已作廢：分期以大表為準，不存在「沒有診斷」這一類人。'
     '本表只保留用藥資料可得性的敏感度對照'),
    ('Table S10 panel genera',      f'TableS10_panel_genera_{ST}.csv',
     '★ 20261030_v65：新增每屬的全樣本 beta／t／p（批次校正模型）。'
     'folds_selected >= 3 共 19 屬，這是菌屬名單的唯一權威來源'),
    ('Table S11 microbiota-clinical', f'TableS11_microbiota_clinical_{ST}.csv',
     '兩種菌相分數對臨床結局；CKM 分期改用大表 V10'),
    ('Table S12 genus-clinical',    f'TableS12_genus_clinical_{ST}.csv',
     '★ 20261030_v65：穩定菌屬名單統一以 X2（TableS10）為準，18 屬改為 19 屬，'
     '熱圖 306 格改為 323 格，BH 全部重算'),
    ('Table S13 abundance-stage',   f'TableS13_abundance_stage_{ST}.csv',
     '★ 20261030_v65：19 個穩定菌屬；分期用大表 V10。沒有任何一屬通過校正'),
    ('Table S17 MetaCyc enrichment', 'TableS17_metacyc_enrichment_20261107_v66.csv',   # 20261107：已撤除的 Gemmiger 暴露列移除
    
     '365 條 MetaCyc 路徑 × 四個 exposure。三種數量分開報：主要模型、NSTI 校正'
     '模型自己、兩者交集（面板 251／115／69；Gemmiger 96／38／32；'
     '醚脂比值 7／1／1；CKM 分期 3 或 4 為 0／0／0）'),
    ('Table S18 MetaCyc vs NSTI',   f'TableS18_metacyc_vs_nsti_20260911_v55.csv',
     '每條路徑對加權 NSTI 本身的關聯'),
    # 20261030_v65 依裁示撤除 Table S19（Gemmiger 定向複製）：他們的是他們的，我們的是自己的發現
    ('Table S20 matched pairs',     f'TableS20_pathway_metabolite_pairs_20260911_v56.csv',
     '事前配對的路徑–代謝物。主要模型 0 條、交集 0 條，但 NSTI 校正模型自己有 6 條'
     '（全與 arginine 有關），見 Table S31'),
    ('Table S21 broad scan',        f'TableS21_pathway_metabolite_broad_20260911_v56.csv',
     '路徑 × 代謝物的廣泛掃描。主要模型 0 條、交集 0 條，校正模型自己有 5 條，'
     '見 Table S31'),
    ('Table S22 metabolite-clinical', f'TableS22_metabolite_clinical_20260911_v56.csv',
     '代謝物對臨床結局'),
    ('Table S23 complete chains',   f'TableS23_complete_chains_20260911_v56.csv',
     '完整鏈（路徑→代謝物→結局，每一環 q<0.05）'),
    ('Table S24 PWY5971 class',     f'TableS24_pwy5971_class_20260911_v56.csv',
     'PWY-5971 對各脂質類別'),
    ('Table S25 PWY5971 joint',     f'TableS25_pwy5971_joint_20260911_v56.csv',
     'PWY-5971 與菌相面板同時進模型'),
    ('Table S26 alpha diversity',   f'TableS26_alpha_diversity_20260913_v60.csv',
     'alpha 多樣性；跨分期檢定改用大表 V10 分期'),
    ('Table S26b alpha adjusted',   f'TableS26b_alpha_adjusted_20260913_v60.csv',
     '校正後的 alpha 多樣性'),
    ('Table S27 PERMANOVA',         f'TableS27_permanova_20260913_v60.csv',
     'beta 多樣性的 PERMANOVA；CKM stage 項改用大表 V10 分期'),
    ('Table S28 panel vs alpha',    f'TableS28_panel_vs_alpha_20260913_v60.csv',
     '面板與群落豐富度同時進模型'),
    ('Table S28b alpha class',      f'TableS28b_alpha_class_specificity_20260913_v60.csv',
     'alpha 指標對各脂質類別的特異性'),
    ('Table S29 nested CV',         f'TableSx_nested_cv_{OUT_ST}.csv',
     '★ 20261030_v65 新增：完整 nested cross-validation。外層切分固定後，前處理、'
     '依 R1 選菌屬、分數標準化與配適全部只在外層訓練集內；信賴區間為整條 pipeline '
     '的 bootstrap（500 次），不是對預測重抽'),
    ('Table S30 holdout 82',        f'TableSy_holdout82_{OUT_ST}.csv',
     '★ 20261030_v65 新增：沒有代謝體的 82 人從未進入任何一次選擇，作為乾淨的'
     '保留集。事件數少，區間寬，只能當方向上的佐證'),
    ('Table S31 adjusted-only pairs', f'TableSz_adjusted_only_pairs_{OUT_ST}.csv',
     '★ 20261030_v65 新增：主要模型未達標、NSTI 校正模型才達 q<0.05 的路徑–代謝物'
     '配對，共 11 條。列出來是為了不把校正後才出現的訊號講成不存在；'
     '這些只能當探索性線索'),
    ('Table S32 delta AUC',         f'TableSw_delta_auc_{OUT_ST}.csv',
     '★ 20261030_v65 新增：配對的 ΔAUC。兩個模型評的是同一批人，AUC 高度相關，'
     '分別看各自的區間會嚴重低估差值的精確度，所以在每一次 bootstrap 重抽裡跑完整'
     ' nested CV 再相減。醚脂比值對三酸甘油酯 +0.039（95% CI 0.013–0.074）；'
     '菌相面板單獨對三個結局的區間都含 0'),
    ('Table S33 score to eight ratios', f'TableSm_panel_to_eight_ratios_{OUT_ST}.csv',
     '★ 20261030_v65 新增：事前三菌屬分數與 CV 面板對八個事前比值各自的效果（Figure 2B）；'
     'BH 在八個之間。只有 R1 過 q < 0.05'),
    ('Table S34 permutation control', f'TableSs_permutation_nce_{OUT_ST}.csv',
     '★ 20261030_v65 新增：置換負控制——把 R1 打亂後用同一條選擇流程重建面板、'
     '再對真的 R1 檢定（Figure 2D）。事前分數不經選擇，不受此影響'),
    ('Table S35 stage boundaries',  'TableS36_boundaries_with_contrast_20261107_v66.csv',
     '★ 20261030_v65 新增：R1 與兩種菌相分數對四個 CKM 分期切點（≥1 到 ≥4）的直接效果，n = 743（Figure 6B）'),
    ('Table S36 eight-ratio block', f'TableSp_eight_ratio_block_{OUT_ST}.csv',
     '★ 20261030_v65 新增：八個比值當一個區塊的概似比檢定，每個切點一列，代謝體樣本與菌相樣本分開'),
    ('Table S37 eight-ratio per ratio', f'TableSo_eight_ratio_per_ratio_{OUT_ST}.csv',
     '★ 20261030_v65 新增：同一聯合模型內各比值的係數；BH 在同一切點的八個之間（Figure 6D）'),
    ('Table S39 omnibus',           f'TableSj_omnibus_{OUT_ST}.csv',
     '★ 20261030_v65 新增：八個比值當一個區塊對菌相分數的 F 檢定與 partial R²'),
    ('Table S40 parallel mediation', f'TableSh_parallel_mediation_{OUT_ST}.csv',
     '★ 20261030_v65 新增：八條平行中介（Figure 6C）；總間接效果與各通路的間接效果，bootstrap 2,000 次'),
    ('Table S41 disease axis',      f'TableSl_metabolite_disease_axis_{OUT_ST}.csv',
     '★ 20261030_v65 新增：從疾病端建立的八比值軸（在不重疊的 2,936 人配適）對菌相分數（Figure 6E）'),
    ('Table S42 axis weights',      f'TableSk_axis_weights_{OUT_ST}.csv',
     '★ 20261030_v65 新增：各切點疾病軸的八個權重（Figure 6E 右）'),
    ('Table S43 microbiota axis',   f'TableSi_microbiota_metabolite_axis_{OUT_ST}.csv',
     '★ 20261030_v65 新增：從菌相端建立的八比值軸對各切點'),
    ('Table S44 single mediation',  f'TableSv_mediation_full_{OUT_ST}.csv',
     '★ 20261030_v65 新增：單一中介（R1）的 a／b／c／c′ 敏感度分析'),
    ('Table S45 temporal',          f'TableSu_temporal_{OUT_ST}.csv',
     '★ 20261030_v65 新增：有 >180 天前另一次採血者（n = 521）的時間敏感度（Figure S10）；只有一次採便，不能說時序'),
    ('Table S47 MetaCyc by taxon',   'TableSmc_metacyc_taxa_20261031_v65.csv',
     '★ 20261031_v65 新增：365 條 MetaCyc 路徑 × 事前分數、三個事前菌屬各自、CV 面板、R1；主要模型與 NSTI 校正模型；'
     'survives = 兩者都 q<0.05；category 為名稱關鍵字分類（只用於作圖）'),
]

HEAD = [
    'Supplementary Tables — A plasma ether-lipid ratio links gut bacterial fatty-acid synthesis to cardiovascular–kidney–metabolic phenotype',
    f'Revision {OUT_ST}. Tables are numbered in order of first citation in the main text; each sheet is listed below with its row and column counts.',
    'CKM stage is assigned at each participant\'s own blood-draw visit (contemporaneous with stool collection) from the cohort master table: stages 0–2 by the health-examination criteria, stage 3 by cardiac, natriuretic-peptide, troponin or renal criteria met on or before that date, stage 4 by a qualifying cardiovascular diagnosis before the draw.',
    'Every statistic was produced by the analysis pipeline from the source result files; Benjamini–Hochberg q values are computed within the family stated for each table.',
    '',
]

def add_family(d):
    """S5：把檢定家族寫在表上，而不是只寫在方法裡（沿用 v61 的作法）。"""
    d = d.copy()
    d.insert(2, 'family', d.group + ' \u00d7 ' + d.score)
    d.insert(3, 'n_in_family', d.groupby('family').species.transform('size'))
    return d


# ---- 20261031_v65：依正文首次引用順序重編（Y8 產出的對照表）；沒被引用的表直接報錯
MAPcsv = pd.read_csv(f'supp_renumber_map_{OUT_ST}.csv')
MAPT = {str(r.old): int(r.new) for _, r in MAPcsv[MAPcsv.kind == 'Table'].iterrows()}
MAPF = {str(r.old): int(r.new) for _, r in MAPcsv[MAPcsv.kind == 'Figure'].iterrows()}
import re as _re
def _remap_note(t):
    t = _re.sub(r'Table S(\d+b?)', lambda m: f'Table S{MAPT[m.group(1)]}' if m.group(1) in MAPT else m.group(0), t)
    t = _re.sub(r'Figure S(\d+b?)', lambda m: f'Figure S{MAPF[m.group(1)]}' if m.group(1) in MAPF else m.group(0), t)
    return t
_new = []
for name, path, note in SHEETS:
    k = _re.match(r'Table S(\d+b?) (.*)', name)
    if k.group(1) not in MAPT:
        raise SystemExit(f'!! {name} 沒有被正文引用，不能放進活頁簿')
    _new.append((MAPT[k.group(1)], f'Table S{MAPT[k.group(1)]} {k.group(2)}', path, _remap_note(note)))
_new.sort()
assert [x[0] for x in _new] == list(range(1, len(_new) + 1)), '活頁簿的表與正文引用的集合不一致'
SHEETS = [(n, p, t) for _, n, p, t in _new]
HEAD = [_remap_note(h) for h in HEAD]

rows, order = [], []
for name, path, note in SHEETS:
    if not os.path.exists(path):
        raise SystemExit(f'!! 找不到 {path}（{name}）')
    d = pd.read_csv(path)
    if 'species' in name and 'TableS5_species' in path:
        d = add_family(d)
    if 'pre-specification' in name:   # 「Where reported」欄以正文 md 的舊編號寫，這裡依對照表改成最終編號
        d = d.map(lambda v: _remap_note(v) if isinstance(v, str) else v)
    order.append((name, d))
    rows.append(dict(table=name, rows=len(d), cols=d.shape[1]))   # 20261031：內部中文註解欄依裁示整欄移除
contents = pd.DataFrame(rows)

out = 'out/SupplementaryTables_20261107_v66.xlsx'
with pd.ExcelWriter(out, engine='openpyxl') as w:
    pd.DataFrame({'Supplementary Tables': HEAD}).to_excel(
        w, sheet_name='Contents', index=False)
    contents.to_excel(w, sheet_name='Contents', index=False, startrow=len(HEAD) + 2)
    for name, d in order:
        d.to_excel(w, sheet_name=name[:31], index=False)

# ---- 稽核：每一張有 p 與 q 的表都不可以出現 q < p
bad = []
for name, d in order:
    if {'p', 'q'} <= set(d.columns):
        n = int((pd.to_numeric(d.q, errors='coerce') <
                 pd.to_numeric(d.p, errors='coerce') - 1e-12).sum())
        if n:
            bad.append((name, n))
print(f'寫出 {out}：{len(order)} 張表 + Contents')
print('★ 全表稽核 q >= p：' + ('通過，沒有任何一列 q < p' if not bad else f'失敗 {bad}'))
assert not bad, bad
print(contents[['table', 'rows', 'cols']].to_string(index=False))
