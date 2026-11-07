# -*- coding: utf-8 -*-
"""CKM 分期一律取自大表 V10（CKM_stage_v3），各人採血當次那一列。 20261010_v64

使用者裁示：「大表上怎麼寫他是第幾期他就是」。

大表 V10 的 `CKM_stage_v3` 以各人自己的 `data exam` 為判定時點：
  stage 4  抽血日之前已有合格心血管診斷（CHD／HF／Stroke／PAD／AF；
           住院 ≥1 次｜急診 ≥1 次｜門診 ≥2 次且間隔 ≥30 天）
  stage 3  抽血日當天或之前符合 LVMI／E'e／TRV／NP／hs-TnT／eGFR<30 任一項
  stage 0–2 仍由世代自己的健檢資料判定，CGRD 不插手
  只往上修、不往下修

管線裡原本每支腳本各自用 OFF(CKM_stage, stage3_criteria_met) 重算，
現在一律改呼叫 master_stage()，確保所有圖表用同一個分期。
大表沒有該訪次判定時（無 exam 日期或嘉義）退回舊式 OFF，並印出筆數。
"""
import numpy as np
import pandas as pd

_LOOKUP = 'stage_v10_lookup_20261010_v64.csv'
_V10 = None


def _load():
    global _V10
    if _V10 is None:
        v = pd.read_csv(_LOOKUP)
        v['subject_id'] = v.subject_id.astype(str)
        v['exam'] = pd.to_datetime(v.exam, errors='coerce')
        _V10 = v[['subject_id', 'exam', 'CKM_stage_v3']].dropna(subset=['exam'])
    return _V10


def master_stage(df, tag='', quiet=False):
    """回傳該資料框每一列的 CKM 分期，取自大表 V10 的採血當次判定。"""
    off = np.where(df.CKM_stage.values == 4, 4,
                   np.where((df.stage3_criteria_met.values >= 1) &
                            (np.isin(df.CKM_stage.values, [1, 2])), 3,
                            df.CKM_stage.values)).astype(float)
    sid = (df.subject_id if 'subject_id' in df.columns else df.index.to_series())
    k = pd.DataFrame({'subject_id': np.asarray(sid).astype(str),
                      'exam': pd.to_datetime(df.exam, errors='coerce').values})
    new = k.merge(_load(), on=['subject_id', 'exam'], how='left').CKM_stage_v3.values
    hit = np.isfinite(new)
    out = np.where(hit, new, off)
    if not quiet:
        print(f'  [大表 V10 分期] {tag}: 對到 {int(hit.sum())}/{len(df)}'
              f'（{100 * hit.mean():.1f}%），與舊式 OFF 不同 {int(np.sum(hit & (new != off)))} 列，'
              f'其餘 {int((~hit).sum())} 列沿用 OFF')
    return out
