# -*- coding: utf-8 -*-
"""祖源主成分敏感度併進穩健度階梯（Table S6 多一列）                 20261031_v65

與 X4 的 PC 敏感度模型完全相同（年齡、性別、BMI、16S 批次 + 10 個遺傳主成分 + 基因型批次；HC3；t 分布 P），
在有主成分的 585 人上估計兩種暴露對 R1 與 log TG。結果追加到 TableS6（pipeline 檔與 out 檔都加），
並與 TableS15 的既有值 assert 一致。輸出 numbers_ancestry_{ST}.csv。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
UP = '/mnt/user-data/uploads/CKM+microbiome'; SRC = '20260911_v54'; ST = '20261030_v65'
rint = lambda v: (lambda a, ok, o: (o.__setitem__(ok, stats.norm.ppf(
        (stats.rankdata(a[ok]) - 0.5) / ok.sum())), o)[1])(
        np.asarray(v, float), np.isfinite(np.asarray(v, float)), np.full(len(np.asarray(v, float)), np.nan))


def hc3(X, y):
    Xi = np.linalg.pinv(X.T @ X); b = Xi @ (X.T @ y); e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, Xi, X); u = e / (1 - np.clip(h, 0, .999))
    return b, np.sqrt(np.diag(Xi @ (X.T @ (X * (u ** 2)[:, None])) @ Xi))


PHE = pd.read_csv(f'pheno_all_{SRC}.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
A0 = pd.read_csv(f'{UP}/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False); A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.is_preferred == True) & (A0.delta_days.abs() <= 180) & (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
S825 = list(A.subject_id); S743 = [s for s in S825 if s in set(PHE.subject_id)]
mb = A.set_index('subject_id'); ph = PHE.set_index('subject_id')
PCS = pd.read_csv(f'{UP}/analysis/pheno_final_20260822_v25.csv'); PCS['subject_id'] = PCS.subject_id.astype(str)
PCS = PCS.set_index('subject_id')[[f'PC{i}' for i in range(1, 11)] + ['gbatch']]
SC = pd.read_csv(f'out/panel_scores_{ST}.csv'); SC['subject_id'] = SC.subject_id.astype(str); SC = SC.set_index('subject_id')
S585 = [s for s in S743 if s in PCS.index]
y1 = pd.Series(rint(ph.R1_PCae_lysoPC_z.values), index=ph.index)
tg = pd.Series(rint(np.log(pd.to_numeric(mb.lab_triglyceride, errors='coerce').reindex(S825))), index=S825)
age = pd.to_numeric(mb.age, errors='coerce'); sex = pd.to_numeric(mb.sex, errors='coerce'); bmi = pd.to_numeric(mb.BMI, errors='coerce')
BAT = pd.get_dummies(mb.batch, drop_first=True).astype(float)


def est(y, x, subj):
    X = np.column_stack([np.asarray(x, float), np.ones(len(subj)), age.reindex(subj).values, sex.reindex(subj).values,
                         bmi.reindex(subj).values, BAT.reindex(subj).values,
                         PCS.reindex(subj)[[f'PC{i}' for i in range(1, 11)]].values,
                         pd.get_dummies(PCS.reindex(subj).gbatch, drop_first=True).astype(float).values])
    yv = np.asarray(y, float); ok = np.isfinite(X).all(1) & np.isfinite(yv); X, yv = X[ok], yv[ok]
    keep = np.r_[True, True, X[:, 2:].std(0) > 1e-10]; X = X[:, keep]
    b, se = hc3(X, yv)
    return b[0], se[0], 2 * stats.t.sf(abs(b[0] / se[0]), ok.sum() - X.shape[1]), int(ok.sum())


S15 = pd.read_csv(f'out/TableS15_pc_sensitivity_{ST}.csv').set_index('result')
rows, num = [], []
for ex, col in [('CV panel', 'panel_oof'), ('Pre-specified score', 'score_prior')]:
    for yn, yv in [('PC ae/lysoPC', y1), ('log(triglycerides)', tg)]:
        b, se, p, n = est(yv.reindex(S585).values, SC[col].reindex(S585).values, S585)
        rows.append(dict(Exposure=ex, Outcome=yn, Model='+ 10 genetic principal components', Beta=b, SE=se, P=p, N=n))
        tag = ('pre' if ex.startswith('Pre') else 'cv') + ('_R1' if yn.startswith('PC') else '_TG')
        for k, v in [('beta', b), ('se', se), ('p', p), ('n', n)]: num.append(dict(key=f'anc_{tag}_{k}', value=v, source='X27'))
# 與 TableS15 核對
chk = {('CV panel', 'PC ae/lysoPC'): 'Microbiota panel → ether-lipid ratio', ('Pre-specified score', 'PC ae/lysoPC'): 'Pre-specified score → ether-lipid ratio',
       ('CV panel', 'log(triglycerides)'): 'Microbiota panel → log(triglycerides)'}
for r in rows:
    k = (r['Exposure'], r['Outcome'])
    if k in chk:
        assert abs(r['Beta'] - S15.loc[chk[k], 'beta_genotyped_with_pcs']) < 1e-9 and abs(r['P'] - S15.loc[chk[k], 'p_genotyped_with_pcs']) < 1e-12, (k, r['Beta'])
NEW = pd.DataFrame(rows)
for f in [f'out/TableS6_robustness_{ST}.csv', f'TableS6_robustness_{SRC}.csv']:
    T = pd.read_csv(f); T = T[T.Model != '+ 10 genetic principal components']
    T = pd.concat([T, NEW], ignore_index=True); T.to_csv(f, index=False)
pd.DataFrame(num).to_csv(f'numbers_ancestry_{ST}.csv', index=False)
print(NEW.round(4).to_string(index=False)); print('與 TableS15 一致；TableS6 已追加四列')
