"""Re-verify the six CKM criterion flags against the v54 numbers file — 20260911.

Identical in method to W11 (v48). Re-run rather than carried over, so that every number
in the v54 manuscript is produced by a v54 script from the source data.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

UP = '/mnt/user-data/uploads/CKM+microbiome'; ST = '20260911_v54'
C = pd.read_csv(f'{UP}/CKM_clinical_for_microbiome_20260819_V6.csv', low_memory=False)
male = C.sex == 1
FG, A1C, TG, HDL = C.lab_glucose_AC, C.lab_HbA1c, C.lab_triglyceride, C.lab_HDL
GLUC = ['med_Metformin', 'med_Insulin', 'med_Sulfonylurea', 'med_DPP4i', 'med_SGLT2i',
        'med_GLP1_RA', 'med_PPARgamma_TZD', 'med_AGI']
BP = ['med_ACEi', 'med_ARB', 'med_CCB', 'med_BetaBlocker', 'med_Thiazide',
      'med_LoopDiuretic', 'med_MRA']
on = lambda cols: C[cols].fillna(0).max(axis=1) > 0
SPEC = {
    'C_adiposity': ((C.BMI >= 23) | (C.waist_cm >= np.where(male, 90, 80)),
                    C.BMI.notna() & C.waist_cm.notna()),
    'C_dysfunc':   (FG.between(100, 125) | A1C.between(5.7, 6.4), FG.notna() & A1C.notna()),
    'C_DM':        (((FG >= 126) | (A1C >= 6.5)) | on(GLUC), FG.notna() & A1C.notna()),
    'C_HTN':       ((C.SBP >= 130) | (C.DBP >= 80) | on(BP), C.SBP.notna() & C.DBP.notna()),
    'C_highTG':    (TG >= 135, C.lab_triglyceride_source.notna()),
    'C_lowHDL':    (HDL < np.where(male, 40, 50), C.lab_HDL_source.notna()),
}
rows = []
for flag, (rule, inputs) in SPEC.items():
    ok = inputs & C[flag].notna() & male.notna()
    o = C.loc[ok, flag].astype(int); p = pd.Series(rule, index=C.index)[ok].astype(int)
    rows.append(dict(flag=flag, n_visits=int(ok.sum()),
                     agreement_pct=round((o == p).mean() * 100, 2)))
R = pd.DataFrame(rows); R.to_csv(f'C_flag_verification_{ST}.csv', index=False)

_ok = C.BMI.notna() & C.waist_cm.notna() & C.C_adiposity.notna() & male.notna()
_rule = (C.BMI >= 23) | (C.waist_cm >= np.where(male, 90, 80))
_d = C.loc[_ok & _rule & (C.C_adiposity == 0)]
new = {'c_flag_agree_min': R.agreement_pct.min(), 'c_flag_agree_max': R.agreement_pct.max(),
       'c_flag_agree_min_excl_adiposity': R[R.flag != 'C_adiposity'].agreement_pct.min(),
       'c_flag_n_verified': int(len(R)), 'n_adip_discord': int(len(_d)),
       'adip_discord_median_bmi': round(float(_d.BMI.median()), 1)}
N = pd.read_csv(f'numbers_{ST}.csv'); N = N[~N.key.isin(new)]
pd.concat([N, pd.DataFrame([{'key': k, 'value': v, 'unit': '', 'source': 'X0'}
                            for k, v in new.items()])], ignore_index=True) \
  .to_csv(f'numbers_{ST}.csv', index=False)
print(R.to_string(index=False)); print(new)
