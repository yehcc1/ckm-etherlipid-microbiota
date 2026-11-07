# -*- coding: utf-8 -*-
"""Table S44：本稿 825 份糞便樣本的公開編號（SRA run / sample / BioSample、批次、公開代碼、是否在 743）— 20261107_v66。
只用公開資料欄位（NTCMRC-S 公開代碼、SRA 編號）；院內識別碼只在記憶體內用來配對，不寫出。"""
import xml.etree.ElementTree as ET, shutil, pandas as pd, openpyxl
from openpyxl.styles import Font
r = ET.parse('/home/claude/ext/sra/sra.xml').getroot()
rows = []
for p in r.findall('EXPERIMENT_PACKAGE'):
    s = p.find('SAMPLE'); at = {a.findtext('TAG'): a.findtext('VALUE') for a in p.iter('SAMPLE_ATTRIBUTE')}
    bs = [e.text for e in s.iter('EXTERNAL_ID') if e.get('namespace') == 'BioSample']
    rows.append(dict(library_id=s.get('alias'), subject_code=at['host_subject_id'], sequencing_batch=at['sequencing_batch'],
                     SRA_run=';'.join(x.get('accession') for x in p.iter('RUN')), SRA_sample=s.get('accession'), BioSample=bs[0]))
S = pd.DataFrame(rows).set_index('library_id')
A0 = pd.read_csv('/mnt/user-data/uploads/CKM+microbiome/analysis/CKM_microbiome_analysis_table_v7.csv', low_memory=False)
A0['subject_id'] = A0.subject_id.astype(str)
A = A0[(A0.sample_class == 'study') & (A0.is_preferred == True) & (A0.delta_days.abs() <= 180) & (~A0.subject_id.str.startswith('J'))].drop_duplicates('subject_id')
PHE = pd.read_csv('pheno_all_20260911_v54.csv'); in743 = A.subject_id.isin(set(PHE.subject_id.astype(str)))
T = S.loc[A.sample_id.values].reset_index(drop=True)
T['in_microbiota_metabolome_set'] = (in743.values).astype(int)
assert len(T) == 825 and T.subject_code.is_unique and T.in_microbiota_metabolome_set.sum() == 743
assert set(T.loc[T.in_microbiota_metabolome_set == 1, 'subject_code']) == set(pd.read_csv('chk/zenodo_codes_in_743_20261107.csv').subject_code)
T = T[['subject_code', 'SRA_run', 'SRA_sample', 'BioSample', 'sequencing_batch', 'in_microbiota_metabolome_set']].sort_values('subject_code')
T.to_csv('TableS44_public_accessions_20261107_v66.csv', index=False)
# 加進 v66 補充表活頁簿
src = 'out/SupplementaryTables_20261107_v66.xlsx'
wb = openpyxl.load_workbook(src)
name = 'Table S44 public accessions'
if name in wb.sheetnames: del wb[name]
ws = wb.create_sheet(name); ws.append(list(T.columns))
for c in ws[1]: c.font = Font(bold=True)
for row in T.itertuples(index=False): ws.append(list(row))
for col, w in zip('ABCDEF', (16, 14, 14, 15, 16, 30)): ws.column_dimensions[col].width = w
C = wb['Contents']
for row in C.iter_rows():
    for c in row:
        if isinstance(c.value, str):
            c.value = c.value.replace('Revision 20261030_v65', 'Revision 20261107_v66')
            c.value = c.value.replace('links gut bacterial fatty-acid synthesis', 'links a gut microbiota score and predicted bacterial fatty-acid synthesis')
names = [C.cell(i, 1).value for i in range(1, C.max_row + 1)]
if name not in names: C.append([name, len(T), T.shape[1]])
wb.save(src); print('Table S44', T.shape, 'sheets', len(wb.sheetnames))
