# -*- coding: utf-8 -*-
"""把 v65 所有結果檔裡要進圖與正文的數字合成一個 key → value 檔      20261030_v65

正文與圖說用 {{key}} 代入，沒有任何數字是打上去的。來源優先序：
本地 numbers_20260911_v54（X2/X3/X4 重跑後最新）＞ out/numbers_20261030_v65（整條鏈）。
本輪新增的表（Sh–Sz）各自抽出關鍵值，key 加前綴。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

ST = '20261030_v65'
rows = []
add = lambda k, v, src: rows.append(dict(key=k, value=v, source=src))

a = pd.read_csv('numbers_20260911_v54.csv').drop_duplicates('key', keep='last').set_index('key')
b = pd.read_csv('out/numbers_20261030_v65.csv').drop_duplicates('key', keep='last').set_index('key')
base = pd.concat([b[~b.index.isin(a.index)], a])
for k, r in base.iterrows():
    add(k, r.value, 'pipeline')
for f, pre in [('numbers_metacyc_20260911_v55.csv', ''), ('numbers_nested_20261030_v65.csv', ''),
               ('numbers_figS4_20261030_v65.csv', ''), ('numbers_pwy5971_20260911_v56.csv', ''),
               ('numbers_pathmet_20260911_v56.csv', ''), ('numbers_diversity_20260913_v60.csv', '')]:
    d = pd.read_csv(f).drop_duplicates('key', keep='last')
    for _, r in d.iterrows():
        add(pre + r.key, r.value, f)

# ---- 八比值對菌相（TableSm）
M = pd.read_csv(f'TableSm_panel_to_eight_ratios_{ST}.csv')
for _, r in M.iterrows():
    tag = r.ratio.split('_')[0]; ex = 'pre' if r.exposure.startswith('Pre') else 'cv'
    add(f'm8_{ex}_{tag}_beta', r.beta, 'TableSm'); add(f'm8_{ex}_{tag}_p', r.p, 'TableSm')
    add(f'm8_{ex}_{tag}_q', r.q_across_8, 'TableSm')
# ---- 綜合檢定（TableSj）
J = pd.read_csv(f'TableSj_omnibus_{ST}.csv')
for _, r in J.iterrows():
    ex = 'pre' if r.exposure.startswith('Pre') else 'cv'
    add(f'omni_{ex}_F', r.stat, 'TableSj'); add(f'omni_{ex}_df1', r.df1, 'TableSj')
    add(f'omni_{ex}_df2', r.df2, 'TableSj'); add(f'omni_{ex}_p', r.p, 'TableSj')
    add(f'omni_{ex}_r2', r.partial_r2, 'TableSj'); add(f'omni_{ex}_n', r.n, 'TableSj')
# ---- 置換（TableSs）
S = pd.read_csv(f'TableSs_permutation_nce_{ST}.csv').iloc[0]
for k in ['observed_beta', 'observed_p', 'null_abs_beta_median', 'null_abs_beta_p95',
          'permutation_p', 'n_permutations', 'n']:
    add(f'perm_{k}', S[k], 'TableSs')
# ---- 分期切點（TableSq）
Q = pd.read_csv(f'TableSq_stage_boundaries_{ST}.csv')
for _, r in Q.iterrows():
    cut = r.boundary.replace('stage >= ', '')
    ex = {'Pre-specified score': 'pre', 'CV panel': 'cv', 'Ether-lipid ratio': 'r1'}[r.exposure]
    add(f'bnd_{ex}_ge{cut}_beta', r.beta, 'TableSq'); add(f'bnd_{ex}_ge{cut}_p', r.p, 'TableSq')
    add(f'bnd_ge{cut}_events', r.n_events, 'TableSq')
# ---- 八比值區塊（TableSp）與各比值（TableSo）
P = pd.read_csv(f'TableSp_eight_ratio_block_{ST}.csv')
for _, r in P.iterrows():
    smp = 'big' if r['sample'].startswith('Metabolome') else 'mb'
    cut = r.boundary.replace('stage >= ', '')
    add(f'blk_{smp}_ge{cut}_lr', r.lr_stat, 'TableSp')
    add(f'blk_{smp}_ge{cut}_p', float(stats.chi2.sf(r.lr_stat, 8)), 'TableSp')
    add(f'blk_{smp}_ge{cut}_n', r.n, 'TableSp'); add(f'blk_{smp}_ge{cut}_events', r.n_events, 'TableSp')
O = pd.read_csv(f'TableSo_eight_ratio_per_ratio_{ST}.csv')
for _, r in O.iterrows():
    smp = 'big' if r['sample'].startswith('Metabolome') else 'mb'
    cut = r.boundary.replace('stage >= ', ''); tag = r.ratio.split(' ')[0]
    add(f'r8_{smp}_ge{cut}_{tag}_beta', r.beta, 'TableSo'); add(f'r8_{smp}_ge{cut}_{tag}_q', r.q, 'TableSo')
# ---- 疾病軸（TableSl）與權重（TableSk）
L = pd.read_csv(f'TableSl_metabolite_disease_axis_{ST}.csv')
for _, r in L.iterrows():
    cut = r.boundary.replace('stage >= ', '')
    t = ('val' if 'validates' in r.test else ('pre' if 'Pre-specified' in r.test else 'cv'))
    add(f'dax_ge{cut}_{t}_beta', r.beta, 'TableSl'); add(f'dax_ge{cut}_{t}_p', r.p, 'TableSl')
    add(f'dax_ge{cut}_{t}_n', r.n, 'TableSl'); add(f'dax_ge{cut}_ntrain', r.n_train, 'TableSl')
K = pd.read_csv(f'TableSk_axis_weights_{ST}.csv')
for _, r in K.iterrows():
    add(f'daxw_ge{r.boundary.replace("stage >= ", "")}_{r.ratio.split("_")[0]}', r.weight, 'TableSk')
# ---- 菌相端建軸（TableSi）
I = pd.read_csv(f'TableSi_microbiota_metabolite_axis_{ST}.csv')
for _, r in I[I.axis != 'weight'].iterrows():
    ex = 'pre' if r.exposure.startswith('Pre') else 'cv'
    oc = r.outcome.replace('CKM stage (5-level)', '5lv').replace('CKM stage >= ', 'ge')
    add(f'max_{ex}_{oc}_beta', r.beta, 'TableSi'); add(f'max_{ex}_{oc}_p', r.p, 'TableSi')
for _, r in I[(I.axis == 'weight') & (I.exposure == 'Pre-specified score')].iterrows():
    add(f'maxw_pre_{r.outcome}', r.beta, 'TableSi')
# ---- 平行多重中介（TableSh）
H = pd.read_csv(f'TableSh_parallel_mediation_{ST}.csv')
for _, r in H.iterrows():
    ex = 'pre' if r.exposure.startswith('Pre') else 'cv'
    oc = r.outcome.replace('CKM stage (5-level)', '5lv').replace('CKM stage >= ', 'ge')
    for k in ['c_total', 'c_prime', 'total_indirect', 'ind_lo', 'ind_hi', 'prop_mediated',
              'prop_lo', 'prop_hi', 'n_boot']:
        add(f'pm_{ex}_{oc}_{k}', r[k], 'TableSh')
    for s_ in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']:
        add(f'pm_{ex}_{oc}_ind_{s_}', r[f'ind_{s_}'], 'TableSh')
# ---- 單一中介（TableSv，敏感度）
V = pd.read_csv(f'TableSv_mediation_full_{ST}.csv')
for _, r in V.iterrows():
    ex = 'pre' if r.exposure.startswith('Pre') else 'cv'
    oc = {'log(triglycerides)': 'tg', 'HbA1c': 'hba1c', 'CKM stage (5-level)': '5lv', 'Diabetes': 'dm'}[r.outcome]
    for k in ['a', 'b', 'c', 'c_prime', 'c_prime_lo', 'c_prime_hi', 'indirect', 'indirect_lo',
              'indirect_hi', 'proportion']:
        add(f'sm_{ex}_{oc}_{k}', r[k], 'TableSv')
# ---- 八比值 AUC（TableSn）與配對 ΔAUC（TableSw）
N8 = pd.read_csv(f'TableSn_eight_ratio_auc_{ST}.csv')
for _, r in N8.iterrows():
    oc = {'Diabetes': 'dm', 'Elevated triglycerides': 'tg', 'CKM stage 3 or 4': 'ckm34',
          'CKM stage 2 or higher': 'ckm2'}[r.endpoint]
    md = {'Age, sex, BMI': 'clin', '+ R1 only': 'r1', '+ all eight ratios': 'eight'}[r.model]
    add(f'auc8_{oc}_{md}', r.auc, 'TableSn'); add(f'auc8_{oc}_{md}_delta', r.delta, 'TableSn')
    add(f'auc8_{oc}_{md}_lo', r.lo, 'TableSn'); add(f'auc8_{oc}_{md}_hi', r.hi, 'TableSn')
    add(f'auc8_{oc}_events', r.n_events, 'TableSn')
W = pd.read_csv(f'TableSw_delta_auc_{ST}.csv')
for _, r in W.iterrows():
    oc = {'Diabetes': 'dm', 'Elevated triglycerides': 'tg', 'CKM stage 3 or 4': 'ckm34'}[r.endpoint]
    md = {'+ microbiota panel': 'panel', '+ ether-lipid ratio': 'r1',
          '+ microbiota panel and ether-lipid ratio': 'both',
          'panel on top of ether-lipid ratio': 'panel_on_r1'}[r.model]
    add(f'dauc_{oc}_{md}', r.delta, 'TableSw'); add(f'dauc_{oc}_{md}_lo', r.lo, 'TableSw')
    add(f'dauc_{oc}_{md}_hi', r.hi, 'TableSw'); add(f'dauc_{oc}_{md}_pr0', r.pr_delta_le_0, 'TableSw')
# ---- 時間（TableSu）與保留集（TableSy）
U = pd.read_csv(f'TableSu_temporal_{ST}.csv')
for _, r in U.iterrows():
    ex = 'pre' if r.exposure.startswith('Pre') else 'cv'
    md = 'now' if 'no lag' in r.model else ('past' if 'earlier' in r.model and 'adjusted' not in r.model else 'nowadj')
    add(f'tmp_{ex}_{md}_beta', r.beta, 'TableSu'); add(f'tmp_{ex}_{md}_p', r.p, 'TableSu'); add(f'tmp_n', r.n, 'TableSu')
Y = pd.read_csv(f'out/TableSy_holdout82_{ST}.csv'); Y = Y[Y.panel_rule == 'p01']
for _, r in Y.iterrows():
    oc = {'Diabetes': 'dm', 'Elevated triglycerides': 'tg', 'CKM stage 3 or 4': 'ckm34'}[r.endpoint]
    md = 'clin' if r.model.startswith('Age') else 'panel'
    add(f'hold_{oc}_{md}', r.auc, 'TableSy'); add(f'hold_{oc}_events', r.events, 'TableSy'); add('hold_n', r.n, 'TableSy')
# ---- 穩健度（S6）與 R1→結局（S8）
R = pd.read_csv('out/TableS6_robustness_20261030_v65.csv'); R = R[R.Outcome.str.contains('PC ae/lysoPC')]
MAP = {'Age, sex, batch, BMI': 'base', '+ 8 metabolic drugs': 'drug8', '+ all 28 medications': 'drug28',
       '+ Firmicutes/Bacteroidota ratio': 'fb', '+ log(triglycerides)': 'tg', 'no BMI adjustment': 'nobmi'}
for _, r in R.iterrows():
    if r.Model not in MAP: continue   # 祖源主成分那一列由 X27 的 numbers_ancestry 檔提供
    ex = 'pre' if r.Exposure.startswith('Pre') else 'cv'
    add(f'rob_{ex}_{MAP[r.Model]}_beta', r.Beta, 'TableS6'); add(f'rob_{ex}_{MAP[r.Model]}_p', r.P, 'TableS6')
    add(f'rob_{ex}_{MAP[r.Model]}_n', r.N, 'TableS6')
S8 = pd.read_csv('out/TableS8_ratio_outcome_20261030_v65.csv')
for _, r in S8.iterrows():
    oc = {'log(triglycerides)': 'tg', 'Diabetes': 'dm', 'CKM stage (5-level)': '5lv', 'HbA1c': 'hba1c',
          'CKM stage >= 3': 'ge3'}[r.outcome]
    add(f's8_{r.ratio}_{oc}_beta', r.beta, 'TableS8'); add(f's8_{r.ratio}_{oc}_q', r.q, 'TableS8')
    add(f's8_{r.ratio}_{oc}_n', r.n, 'TableS8')
# ---- 面板 vs 豐富度（S28）
A28 = pd.read_csv('out/TableS28_panel_vs_alpha_20261030_v65.csv')
r_ = A28[(A28.model == 'Pre-specified score + Observed ASVs + depth')]
add('rich_pre_score_beta', float(r_[r_.term == 'Pre-specified score'].beta.iloc[0]), 'TableS28')
add('rich_pre_score_p', float(r_[r_.term == 'Pre-specified score'].p.iloc[0]), 'TableS28')
add('rich_pre_asv_p', float(r_[r_.term == 'Observed ASVs'].p.iloc[0]), 'TableS28')
# ---- 基因型子群（S15）
S15 = pd.read_csv('out/TableS15_pc_sensitivity_20261030_v65.csv')
r_ = S15[S15.result.str.startswith('Pre-specified')].iloc[0]
add('geno_pre_beta_pcs', r_.beta_genotyped_with_pcs, 'TableS15'); add('geno_pre_p_pcs', r_.p_genotyped_with_pcs, 'TableS15')
add('geno_n', r_.n_genotyped, 'TableS15')
# ---- 分期人數
T1 = pd.read_csv('out/Table1_cohort_20261030_v65.csv').iloc[0]
for i in range(5): add(f'stage{i}_n', int(float(T1[f'Stage {i}'])), 'Table1')

out = pd.DataFrame(rows).drop_duplicates('key', keep='last')
out.to_csv(f'numbers_v65_all_{ST}.csv', index=False)
print(f'{len(out)} keys → numbers_v65_all_{ST}.csv')
