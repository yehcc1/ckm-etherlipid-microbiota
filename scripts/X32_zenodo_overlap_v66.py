# -*- coding: utf-8 -*-
"""本稿代謝體樣本與公開 Zenodo p180 資料（10.5281/zenodo.23320220）的逐筆比對 — 20261107_v66。

只用代謝物數值與採樣年份配對，不讀出、不輸出任何個人識別欄位；輸出只有人數。
配對法：每一筆採血取 30 個測得最完整的代謝物，四捨五入到 3 位有效數字組成指紋；
指紋完全相同即視為同一筆。再檢查配對到的年份是否一致、同一受試者的多次採血是否落在同一個 Zenodo 代碼。
"""
import numpy as np, pandas as pd
OURS = '/mnt/user-data/uploads/CKM+microbiome/analysis/04_data_matrices/p180_all.csv'
ZEN = '/home/claude/ext/zenodo/p180_concentrations_20261012_V1.csv'
O = pd.read_csv(OURS, low_memory=False)
Z = pd.read_csv(ZEN, low_memory=False)
mets = [c for c in Z.columns[5:] if c in O.columns]
def num(df): return df[mets].apply(pd.to_numeric, errors='coerce')
On, Zn = num(O), num(Z)
use = (On.notna().mean() + Zn.notna().mean()).sort_values(ascending=False).index[:30].tolist()
def sig3(x):
    x = np.asarray(x, float); out = np.full(x.shape, np.nan)
    ok = np.isfinite(x) & (x != 0)
    e = np.floor(np.log10(np.abs(x[ok]))) - 2
    out[ok] = np.round(x[ok] / 10 ** e) * 10 ** e
    out[np.isfinite(x) & (x == 0)] = 0.0
    return out
def fp(df):
    a = np.column_stack([sig3(df[c].values) for c in use])
    return ['|'.join('' if np.isnan(v) else f'{v:.6g}' for v in row) for row in a]
O['fp'] = fp(On); Z['fp'] = fp(Zn)
zmap = Z.groupby('fp').agg(code=('subject_code', 'first'), n=('subject_code', 'size'), year=('sampling_year', 'first'))
dup_z = int((zmap.n > 1).sum())
O['code'] = O.fp.map(zmap.code); O['zyear'] = O.fp.map(zmap.year)
oy = pd.to_numeric(O.test_year, errors='coerce')
m = O.code.notna()
print(f'本稿 p180：{len(O)} 筆採血、{O.subject_base.nunique()} 人；Zenodo：{len(Z)} 筆、{Z.subject_code.nunique()} 人；比對用代謝物 {len(use)} 個')
print(f'指紋配對成功 {int(m.sum())} 筆；Zenodo 指紋重複 {dup_z} 組；年份一致 {int((oy[m] == O.zyear[m]).sum())} / {int(m.sum())}')
per = O[m].groupby('subject_base').code.nunique()
print(f'同一受試者多筆採血對到不只一個 Zenodo 代碼：{int((per > 1).sum())} 人')
code_of = O[m].groupby('subject_base').code.first()
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
A0 = pd.read_csv('/mnt/user-data/uploads/CKM+microbiome/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) & (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
S825 = list(A.subject_id); S743 = [s for s in S825 if s in set(PHE.subject_id)]
C = pd.read_csv('/mnt/user-data/uploads/CKM+microbiome/CKM_clinical_for_microbiome_20260819_V6.csv', low_memory=False)
C['subject_id'] = C.subject_id.astype(str); SM_ = [s for s in PHE.subject_id if s in set(C.subject_id)]
have = set(code_of.index.astype(str))
rows = []
for nm, ids in [('本稿全部代謝體 3,688', list(PHE.subject_id)), ('ratio–outcome 分析集', SM_), ('菌相∩代謝體 743', S743)]:
    k = sum(1 for s in ids if s in have); rows.append(dict(set=nm, n=len(ids), in_zenodo=k, missing=len(ids) - k))
R = pd.DataFrame(rows); print(R.to_string(index=False))
R.to_csv('chk/zenodo_overlap_counts_20261107.csv', index=False)
# 供菌相端比對：只存「本稿 743 人中有哪些 Zenodo 代碼」，不存任何院內識別碼
pd.Series(sorted(code_of[code_of.index.isin(S743)].values)).to_csv('chk/zenodo_codes_in_743_20261107.csv', index=False, header=['subject_code'])
pd.Series(sorted(code_of[code_of.index.isin(S825)].values)).to_csv('chk/zenodo_codes_in_825_20261107.csv', index=False, header=['subject_code'])
