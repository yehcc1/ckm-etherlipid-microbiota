# -*- coding: utf-8 -*-
"""R1 在各分期切點的係數是否真的不同：同一批人的配對 bootstrap — 20261107_v66。

外審（v66 審稿意見）：「≥2 顯著、≥3 不顯著」本身不構成斜率不同；同一批受試者的切點係數要直接比較。
做法：MB743 內，每個切點以 logistic 模型（R1 + 年齡、性別、BMI、16S 批次，與 Table S36 同一設定）估 R1 係數，
先確認與 Table S36（TableSq）逐位相同；再以受試者為單位重抽 2,000 次，每次四個切點一起重估，
得到 β(≥2) − β(≥k) 的百分位 95% 區間與雙尾 bootstrap P。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, statsmodels.api as sm
exec(open('X23_multipath_v65.py').read().split('def rank_clean')[0])
SQ = pd.read_csv('TableSq_stage_boundaries_20261030_v65.csv')
r1 = pd.Series(rint(PHE['R1_PCae_lysoPC_z'].values), index=PHE.subject_id).reindex(S743).values   # rank-INT over all 3,688, as in Table S36
X = np.column_stack([r1, COV])
CUT = [1, 2, 3, 4]
def coefs(idx):
    out = []
    for c in CUT:
        y = (stage[idx] >= c).astype(float); Xi = X[idx]
        ok = np.isfinite(Xi).all(1)
        Xd = sm.add_constant(Xi[ok][:, np.nanstd(Xi[ok], 0) > 1e-12], has_constant='add')
        out.append(sm.GLM(y[ok], Xd, family=sm.families.Binomial()).fit().params[1])
    return np.array(out)
full = coefs(np.arange(len(S743)))
for c, b in zip(CUT, full):
    t = SQ[(SQ.boundary == f'stage >= {c}') & (SQ.exposure == 'Ether-lipid ratio')].beta.iloc[0]
    assert abs(b - t) < 1e-6, f'>= {c}: {b} vs Table S36 {t}'
print('四個切點的 R1 係數與 Table S36 相同：', np.round(full, 4))
rng = np.random.default_rng(20261107); B = 2000; boot = []
while len(boot) < B:
    idx = rng.integers(0, len(S743), len(S743))
    try: boot.append(coefs(idx))
    except Exception: continue
boot = np.array(boot)
rows = []
for j, c in enumerate(CUT):
    if c == 2: continue
    d = full[1] - full[j]; db = boot[:, 1] - boot[:, j]
    lo, hi = np.percentile(db, [2.5, 97.5])
    p = min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))
    rows.append(dict(contrast=f'beta(stage >= 2) - beta(stage >= {c})', diff=d, lo=lo, hi=hi, p_boot=p, n=len(S743), n_boot=B))
T = pd.DataFrame(rows); T.to_csv('TableSt_boundary_contrast_20261107_v66.csv', index=False)
print(T.round(4).to_string(index=False))
num = []
for _, r in T.iterrows():
    k = r.contrast.split('>= ')[-1].rstrip(')')
    for f in ('diff', 'lo', 'hi', 'p_boot'): num.append(dict(key=f'bc_r1_2v{k}_{f}', value=r[f], source='X31'))
num.append(dict(key='bc_n_boot', value=B, source='X31'))
pd.DataFrame(num).to_csv('numbers_boundary_contrast_20261107_v66.csv', index=False)
