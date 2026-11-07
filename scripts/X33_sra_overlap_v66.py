# -*- coding: utf-8 -*-
"""本稿 825 份糞便樣本是否已在 PRJNA1542453 公開，以及其 host_subject_id 是否與 Zenodo p180 代碼一致 — 20261107_v66。
只輸出人數；不輸出院內識別碼。"""
import pandas as pd
exec(open('X32_zenodo_overlap_v66.py').read().split("print(f'本稿")[0])
code_of = O[m].groupby('subject_base').code.first()
S = pd.read_csv('/home/claude/ext/sra/sra_samples.csv', dtype=str)
A0 = pd.read_csv('/mnt/user-data/uploads/CKM+microbiome/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) & (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
lib = S.set_index('library_id')
A = A.assign(in_sra=A.sample_id.isin(lib.index), sra_code=A.sample_id.map(lib.host_subject_id),
             srr=A.sample_id.map(lib.srr), zen_code=A.subject_id.map(code_of))
A['in743'] = A.subject_id.isin(set(PHE.subject_id))
both = A[A.in743 & A.in_sra]
out = dict(n825=len(A), n825_in_sra=int(A.in_sra.sum()), n743=int(A.in743.sum()), n743_in_sra=int(both.shape[0]),
           n743_code_agree=int((both.sra_code == both.zen_code).sum()),
           sra_samples=len(S), sra_participants=S.host_subject_id.nunique(),
           all_study_samples_in_sra=int(A0[A0.sample_class == 'study'].sample_id.isin(lib.index).sum()),
           all_study_samples=int((A0.sample_class == 'study').sum()))
for k, v in out.items(): print(k, v)
pd.Series(out).to_csv('chk/sra_overlap_counts_20261107.csv', header=['value'])
# 公開代碼清單（只有公開代碼與公開 SRR 編號，可放補充表讓讀者重建 825 / 743）
A[['sra_code', 'srr', 'in743']].rename(columns={'sra_code': 'subject_code', 'srr': 'SRA_run'}).sort_values('subject_code') \
  .to_csv('chk/accession_list_825_20261107.csv', index=False)
# ---- 數字檔（正文 token 用）
Rz = pd.read_csv('chk/zenodo_overlap_counts_20261107.csv')
num = dict(dep_sra_samples=out['sra_samples'], dep_sra_participants=out['sra_participants'],
           dep_825_in_sra=out['n825_in_sra'], dep_743_code_agree=out['n743_code_agree'],
           dep_zen_draws=len(Z), dep_zen_participants=Z.subject_code.nunique(),
           dep_3679_in_zen=int(Rz.in_zenodo[1]), dep_3688_in_zen=int(Rz.in_zenodo[0]), dep_743_in_zen=int(Rz.in_zenodo[2]))
assert num['dep_825_in_sra'] == 825 and num['dep_743_code_agree'] == 743 == num['dep_743_in_zen']
pd.DataFrame([dict(key=k, value=v, source='X32/X33') for k, v in num.items()]).to_csv('numbers_public_deposit_20261107_v66.csv', index=False)
print(num)
