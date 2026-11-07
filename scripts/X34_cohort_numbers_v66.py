# -*- coding: utf-8 -*-
"""Methods 收案段的數字：社區收案比例、每人一份樣本 — 20261107_v66。"""
import pandas as pd
A0 = pd.read_csv('/mnt/user-data/uploads/CKM+microbiome/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
st = A0[A0.sample_class == 'study']
A = st[(st.is_preferred == True) & (st.delta_days.abs() <= 180) & (~st.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
assert len(A) == 825
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); PHE['subject_id'] = PHE.subject_id.astype(str)
C = pd.read_csv('/mnt/user-data/uploads/CKM+microbiome/CKM_clinical_for_microbiome_20260819_V6.csv', low_memory=False)
C['subject_id'] = C.subject_id.astype(str); C = C.drop_duplicates('subject_id').set_index('subject_id')
SM = [s for s in PHE.subject_id if s in C.index]; assert len(SM) == 3679
rs = C.loc[SM, 'recruitment_source']
num = dict(coh_study_samples=len(st), coh_comm_825=(A.recruitment_source == 'Community_Keelung').mean(),
           coh_comm_3679=(rs == 'Community_Keelung').mean(), coh_rs_missing_3679=int(rs.isna().sum()))
print(num); print(rs.value_counts(dropna=False).to_dict())
pd.DataFrame([dict(key=k, value=v, source='X34') for k, v in num.items()]).to_csv('numbers_cohort_20261107_v66.csv', index=False)
