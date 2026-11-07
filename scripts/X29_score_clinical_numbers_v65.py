"""事前分數對臨床結局（Table 2／Table S11 的 MB825 列）→ numbers 檔，供正文引用 — 20261031_v65。"""
import pandas as pd
ST = '20261030_v65'
T = pd.read_csv(f'out/TableS11_microbiota_clinical_{ST}.csv')
T = T[(T.exposure == 'Pre-specified score') & (T.set == 'MB825')].set_index('outcome')
key = {'log(triglycerides)': 'tg', 'HbA1c': 'hba1c', 'Diabetes': 'dm', 'CKM stage (5-level)': 'ckm5', 'CKM stage >= 3': 'ge3'}
rows = []
for o, k in key.items():
    r = T.loc[o]
    for c in ('beta', 'p', 'q', 'n'):
        rows.append(dict(key=f's11_pre_{k}_{c}', value=r[c], source='TableS11'))
T2 = pd.read_csv(f'out/Table2_effects_{ST}.csv')
chk = T2[(T2.Exposure == 'Pre-specified score') & (T2.Outcome == 'Log triglycerides')].iloc[0]
assert abs(float(chk.P) - T.loc['log(triglycerides)', 'p']) < 1e-12, 'Table 2 與 Table S11 的事前分數列不一致'
pd.DataFrame(rows).to_csv('numbers_score_clinical_20261031_v65.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
