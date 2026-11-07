# -*- coding: utf-8 -*-
"""讓 X2 成為菌屬選擇的唯一來源      20261030_v65

背景：X2 與 X3 各自跑了一次 5-fold 的菌屬選擇。seed 與 fold 完全相同，但

  X2：共變數含定序批次（cov_for 會接上 BAT），p 值用 t 分布
  X3：共變數只有 intercept/age/sex/BMI，沒有批次，p 值用常態近似

所以 TableS10 以 folds_selected >= 3 得到 19 屬，X3 的 STABLE 得到 18 屬，
差 Gemmiger／NK4A214 group／Tyzzerella／UCG-005 進、Fusobacterium／
Succinivibrio／Mitsuokella 出。這與先前「八支腳本各算一次 CKM 分期」
是同一類錯誤：同一個量被算了兩次。

實際用在所有模型裡的 panel 分數出自 X2，所以 X2 是權威來源。這支腳本把
X2 改成額外輸出每一屬的全樣本統計與各 fold 的 t 矩陣，X3 之後只讀不算。
批次校正本來就該留著，所以統一之後的名單也是比較對的那一份。
"""
from __future__ import annotations
import sys

SRC = 'X2_core_825_20260911_v54.py'

OLD = '''def select(train_idx, y, subjects):
    """per-genus adjusted association within the training rows; P<0.01, sign weights."""
    idx = [subjects[i] for i in train_idx]
    Cv = cov_for(idx); yv = np.asarray(y, float)[train_idx]
    w = np.zeros(Z.shape[1])
    for j, g in enumerate(Z.columns):
        v = Z[g].reindex(idx).values
        X = np.column_stack([v, Cv]); ok = np.isfinite(X).all(1) & np.isfinite(yv)
        if ok.sum() < 30: continue
        Xi = np.linalg.pinv(X[ok].T @ X[ok]); b = Xi @ (X[ok].T @ yv[ok])
        e = yv[ok] - X[ok] @ b
        se = np.sqrt((e @ e / (ok.sum() - X.shape[1])) * Xi[0, 0])
        p = 2 * stats.t.sf(abs(b[0] / se), ok.sum() - X.shape[1])
        if p < 0.01: w[j] = np.sign(b[0])
    return w
'''

NEW = '''def select_stats(train_idx, y, subjects):
    """每一屬在訓練列上的校正關聯：回傳 beta、t、p。

    共變數取自 cov_for，含定序批次；p 用 t 分布（df = n - k）。
    20261030_v65：原本只回傳 sign 權重，X3 因此自己又算了一次而且少了批次，
    兩邊的 >=3 fold 名單才會一個 19 屬一個 18 屬。改成把統計一起交出去。"""
    idx = [subjects[i] for i in train_idx]
    Cv = cov_for(idx); yv = np.asarray(y, float)[train_idx]
    nb = Z.shape[1]
    beta = np.full(nb, np.nan); tt = np.full(nb, np.nan); pp = np.full(nb, np.nan)
    for j, g in enumerate(Z.columns):
        v = Z[g].reindex(idx).values
        X = np.column_stack([v, Cv]); ok = np.isfinite(X).all(1) & np.isfinite(yv)
        if ok.sum() < 30: continue
        Xi = np.linalg.pinv(X[ok].T @ X[ok]); b = Xi @ (X[ok].T @ yv[ok])
        e = yv[ok] - X[ok] @ b
        se = np.sqrt((e @ e / (ok.sum() - X.shape[1])) * Xi[0, 0])
        beta[j] = b[0]; tt[j] = b[0] / se
        pp[j] = 2 * stats.t.sf(abs(tt[j]), ok.sum() - X.shape[1])
    return beta, tt, pp


def select(train_idx, y, subjects):
    """P<0.01 的 sign 權重，直接由 select_stats 導出。"""
    beta, _, pp = select_stats(train_idx, y, subjects)
    w = np.zeros(Z.shape[1])
    hit = np.isfinite(pp) & (pp < 0.01)
    w[hit] = np.sign(beta[hit])
    return w
'''

OLD2 = '''PANG = pd.DataFrame(dict(genus=Z.columns, weight_full=W_full,
                         folds_selected=sel_count.values))
PANG[PANG.folds_selected > 0].sort_values(['folds_selected', 'genus'], ascending=[False, True]) \\
    .to_csv(f'TableS10_panel_genera_{ST}.csv', index=False)
'''

NEW2 = '''# 20261030_v65：把每一屬的全樣本統計與各 fold 的 t 一起留下來，X3 之後只讀不算。
FOLD_T = []
for tr, _te in KFold(5, shuffle=True, random_state=20260822).split(np.arange(len(S743))):
    FOLD_T.append(select_stats(tr, y1_743, S743)[1])
FOLD_T = np.vstack(FOLD_T)
B_full, T_full, P_full = select_stats(np.arange(len(S743)), y1_743, S743)
np.savez(f'panel_fold_stats_{ST}.npz', genus=np.array(list(Z.columns), dtype=object),
         fold_t=FOLD_T, t_full=T_full, beta_full=B_full, p_full=P_full,
         folds_selected=sel_count.values)
assert ((np.isfinite(FOLD_T) & (2 * stats.t.sf(np.abs(FOLD_T), 1e6) < 0.01)).sum(0) >= 0).all()

PANG = pd.DataFrame(dict(genus=Z.columns, weight_full=W_full, beta_full=B_full,
                         t_full=T_full, p_full=P_full, folds_selected=sel_count.values))
PANG[PANG.folds_selected > 0].sort_values(['folds_selected', 'genus'], ascending=[False, True]) \\
    .to_csv(f'TableS10_panel_genera_{ST}.csv', index=False)
# 全樣本權重的方向必須與全樣本 beta 一致，否則兩套統計已經不是同一個模型
_chk = (W_full != 0)
assert np.array_equal(np.sign(B_full[_chk]), W_full[_chk]), '全樣本權重與 beta 方向不一致'
'''

s = open(SRC, encoding='utf-8').read()
for old, new, why in [(OLD, NEW, 'select 改為回傳統計'),
                      (OLD2, NEW2, 'TableS10 增列統計並輸出 fold t 矩陣')]:
    if s.count(old) != 1:
        sys.exit(f'!! {why}：片段出現 {s.count(old)} 次（預期 1）')
    s = s.replace(old, new)
open(SRC, 'w', encoding='utf-8').write(s)
print('X2 已改：select_stats 產生 beta/t/p；TableS10 增列 beta_full/t_full/p_full；'
      '另存 panel_fold_stats_*.npz')
