"""Presentation versions of main Tables 2 and 3 — 20260911_v54.

Table 2  every microbiota exposure against the ether-lipid ratio and against each clinical
         outcome, in the set in which that model is estimable, with both scores side by side
Table 3  cross-validated discrimination of clinical state, against the clinical baseline

Both are assembled from the result files written by X2 and X3; no value is typed here.
Table 2 is cross-checked against Table S6 so the two can never drift apart.
"""
import numpy as np, pandas as pd

ST = '20260911_v54'
ROB = pd.read_csv(f'TableS6_robustness_{ST}.csv')
MC = pd.read_csv(f'TableS11_microbiota_clinical_{ST}.csv')
CLS = pd.read_csv(f'TableS4_class_specificity_{ST}.csv')
DISC = pd.read_csv(f'Table3_discrimination_{ST}.csv')

ci = lambda b, se: f'{b:+.3f} ({b-1.96*se:+.3f} to {b+1.96*se:+.3f})'.replace('-', '−')

rows = []
base = ROB[ROB.Model == 'Age, sex, batch, BMI']
for exp_ in ['Pre-specified score', 'CV panel']:
    r = base[(base.Exposure == exp_) & (base.Outcome == 'PC ae/lysoPC')].iloc[0]
    rows.append(dict(Exposure=exp_, Outcome='Ether-lipid ratio (PC ae / lysoPC)',
                     Set='Metabolomics + 16S', N=int(r.N), Estimate=ci(r.Beta, r.SE),
                     P=r.P, Q=''))
for exp_ in ['Pre-specified score', 'CV panel']:
    r = CLS[(CLS.score == exp_) & (CLS.phenotype == 'PC ae / PC aa')].iloc[0]
    rows.append(dict(Exposure=exp_, Outcome='PC ae / PC aa ratio', Set='Metabolomics + 16S',
                     N=int(r.n), Estimate=ci(r.beta, r.se), P=r.p, Q=''))
OUT = {'log(triglycerides)': 'Log triglycerides', 'HbA1c': 'HbA1c',
       'Diabetes': 'Diabetes (log-odds)', 'CKM stage (5-level)': 'CKM stage (5-level)',
       'CKM stage >= 3': 'CKM stage 3 or 4 (log-odds)'}
for exp_ in ['Pre-specified score', 'CV panel']:
    for o, lab in OUT.items():
        r = MC[(MC.exposure == exp_) & (MC.outcome == o) & (MC['set'] == 'MB825')].iloc[0]
        rows.append(dict(Exposure=exp_, Outcome=lab, Set='All 16S profiled', N=int(r.n),
                         Estimate=ci(r.beta, r.se), P=r.p, Q=r.q))
T2 = pd.DataFrame(rows)
T2['Exposure'] = T2.Exposure.replace({'CV panel': 'Cross-validated panel'})
T2.to_csv(f'Table2_effects_{ST}.csv', index=False)

# cross-check: every ether-lipid row must equal its Table S6 base model exactly
for _, r in T2[T2.Outcome.str.startswith('Ether-lipid')].iterrows():
    e = 'CV panel' if r.Exposure.startswith('Cross') else r.Exposure
    s6 = ROB[(ROB.Exposure == e) & (ROB.Outcome == 'PC ae/lysoPC') &
             (ROB.Model == 'Age, sex, batch, BMI')].iloc[0]
    assert abs(float(r.P) - float(s6.P)) < 1e-12 and int(r.N) == int(s6.N), \
        f'Table 2 has drifted from Table S6 for {r.Exposure}'

D = DISC.copy()
D['Model'] = D.model
D['Endpoint'] = D.endpoint
D['Set'] = D['set'].map({'MB825': 'All 16S profiled', 'MB743': 'Metabolomics + 16S'})
D['Events / n'] = D.events.astype(str) + ' / ' + D.n.astype(str)
D['AUC (95% CI)'] = D.apply(lambda r: f'{r.auc:.3f} ({r.lo:.3f} to {r.hi:.3f})', axis=1)
for c in ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv']:
    D[c.capitalize()] = D[c].map(lambda v: f'{v:.2f}')
D = D[['Set', 'Endpoint', 'Events / n', 'Model', 'AUC (95% CI)', 'Accuracy',
       'Sensitivity', 'Specificity', 'Ppv', 'Npv']].rename(columns={'Ppv': 'PPV', 'Npv': 'NPV'})
D.to_csv(f'Table3_discrimination_presented_{ST}.csv', index=False)

print(T2.to_string(index=False)); print(); print(D.to_string(index=False))
