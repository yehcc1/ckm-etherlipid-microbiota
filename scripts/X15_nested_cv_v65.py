# -*- coding: utf-8 -*-
"""完整 nested CV 與 82 人保留集的執行腳本      20261030_v65

資料與函式在 nested_core_v65.py，X17 畫 Figure 5 時用同一套。
"""
from nested_core_v65 import *            # noqa: F401,F403

# ============================== A. nested CV ==============================
rows = []
for rule in ['p01', 'topk']:
    for setname, subj in [('MB825', S825), ('MB743', S743)]:
        for en, y in END.items():
            preds, yv, kk = nested_predictions(subj, y, rule)
            for mn in MODELS:
                if setname == 'MB825' and 'r1' in MODELS[mn]: continue
                a = auc(yv, preds[mn])
                n = int((np.isfinite(preds[mn]) & np.isfinite(yv)).sum())
                ev = int(np.nansum(yv[np.isfinite(preds[mn])]))
                rows.append(dict(panel_rule=rule, set=setname, endpoint=en, model=mn,
                                 auc=a, n=n, events=ev,
                                 k_per_fold='' if rule == 'p01' else ';'.join(map(str, kk))))
            print(f'  {rule:5s} {setname} {en[:24]:24s} '
                  + '  '.join(f'{auc(yv, preds[m]):.3f}' for m in MODELS
                              if not (setname == "MB825" and "r1" in MODELS[m])))
NEST = pd.DataFrame(rows)

# ----------------------- bootstrap：整條 pipeline -----------------------
print(f'\nbootstrap {B_BOOT} 次（整條 pipeline，含菌屬選擇）…')
rng = np.random.default_rng(20261030)
boot = {(r, s, e, m): [] for r in ['p01'] for s in ['MB743'] for e in END for m in MODELS}
for b in range(B_BOOT):
    take = list(pd.Series(S743).sample(len(S743), replace=True, random_state=int(rng.integers(1e9))))
    for en, y in END.items():
        try:
            preds, yv, _ = nested_predictions(take, y, 'p01', seed=SEED_OUTER + b)
        except Exception:
            continue
        for mn in MODELS:
            boot[('p01', 'MB743', en, mn)].append(auc(yv, preds[mn]))
    if (b + 1) % 50 == 0: print(f'   {b + 1}/{B_BOOT}')

ci = []
for key, vals in boot.items():
    v = np.array([x for x in vals if np.isfinite(x)])
    if len(v) < 20: continue
    lo, hi = np.percentile(v, [2.5, 97.5])
    ci.append(dict(panel_rule=key[0], set=key[1], endpoint=key[2], model=key[3],
                   boot_lo=lo, boot_hi=hi, boot_n=len(v)))
CI = pd.DataFrame(ci)
NEST = NEST.merge(CI, on=['panel_rule', 'set', 'endpoint', 'model'], how='left')
NEST.to_csv(f'TableSx_nested_cv_{ST}.csv', index=False)

# ===================== B. 82 人保留集 =====================
hold = []
mu, sd = RAWCLR.reindex(S743).mean(), RAWCLR.reindex(S743).std()
Zc = (RAWCLR - mu) / sd
beta, p, tt = select_stats(S743, Zc)
for rule, kk in [('p01', None), ('topk', 40)]:
    w = panel_weights(beta, p, tt, rule, kk)
    raw_tr = Zc.reindex(S743).values @ w
    m_, s_ = np.nanmean(raw_tr), (np.nanstd(raw_tr) or 1.0)
    sc_tr = (raw_tr - m_) / s_
    sc_te = (Zc.reindex(S82).values @ w - m_) / s_
    for en, y in END.items():
        for mn, cols in MODELS.items():
            if 'r1' in cols: continue        # 82 人沒有代謝體
            pr = fit_predict(S743, S82, cols, sc_tr, sc_te, y)
            yv = y.reindex(S82).values.astype(float)
            ok = np.isfinite(pr) & np.isfinite(yv)
            hold.append(dict(panel_rule=rule, endpoint=en, model=mn,
                             auc=auc(yv, pr), n=int(ok.sum()), events=int(np.nansum(yv[ok])),
                             n_panel_genera=int((w != 0).sum())))
HOLD = pd.DataFrame(hold)
HOLD.to_csv(f'TableSy_holdout82_{ST}.csv', index=False)

print('\n=== nested CV（MB743，P<0.01 panel）===')
print(NEST[(NEST.set == 'MB743') & (NEST.panel_rule == 'p01')]
      [['endpoint', 'model', 'auc', 'boot_lo', 'boot_hi', 'n', 'events']]
      .round(3).to_string(index=False))
print('\n=== 82 人保留集 ===')
print(HOLD.round(3).to_string(index=False))

F = []
for en in END:
    d = NEST[(NEST.set == 'MB743') & (NEST.panel_rule == 'p01') & (NEST.endpoint == en)]
    base = float(d[d.model == 'Age, sex, BMI'].auc.iloc[0])
    pan = float(d[d.model == '+ microbiota panel'].auc.iloc[0])
    r1m = float(d[d.model == '+ ether-lipid ratio'].auc.iloc[0])
    tag = en.lower().replace(' ', '_').replace('3_or_4', '34')
    F += [dict(key=f'nested_{tag}_auc_clinical', value=base, unit='', source='nested CV'),
          dict(key=f'nested_{tag}_auc_panel', value=pan, unit='', source='nested CV'),
          dict(key=f'nested_{tag}_auc_r1', value=r1m, unit='', source='nested CV'),
          dict(key=f'nested_{tag}_gain_panel', value=pan - base, unit='', source='nested CV'),
          dict(key=f'nested_{tag}_gain_r1', value=r1m - base, unit='', source='nested CV')]
pd.DataFrame(F).to_csv(f'numbers_nested_{ST}.csv', index=False)
print(f'\n寫出 TableSx_nested_cv_{ST}.csv、TableSy_holdout82_{ST}.csv、numbers_nested_{ST}.csv')
