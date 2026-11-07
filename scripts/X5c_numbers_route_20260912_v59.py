"""Numbers for the Figure 3D route map — 20260912_v59.

Starts from the v58 numbers file and adds one key per node of the route, read from the
MetaCyc enrichment table. beta is from the primary model and q* from the coverage-adjusted
model, the same convention as Figure S5C and Figure 3D. Each pathway is checked to survive
both models before its numbers are written, so a key can never carry a value the figure
would not draw.
"""
import pandas as pd, numpy as np

M = pd.read_csv('TableS17_metacyc_enrichment_20260911_v55.csv')
P = M[M.exposure == 'Microbiota panel'].set_index('pathway')
N = pd.read_csv('numbers_20260911_v58.csv')

ROUTE = {'sucroseIV': 'PWY-5384', 'sucroseIII': 'PWY-621', 'bifshunt': 'P124-PWY',
         'mixedacid': 'FERMENTATION-PWY', 'heterolactic': 'P122-PWY', 'acetone': 'PWY-6588',
         'palmitate': 'PWY-5971',
         'glutdeg': 'P162-PWY', 'gabadeg': 'PWY-5022', 'hisdeg': 'HISDEG-PWY',
         'tca7': 'PWY-7254', 'propanoate': 'P108-PWY', 'urea': 'PWY-4984'}
POS = ['sucroseIV', 'sucroseIII', 'bifshunt', 'mixedacid', 'heterolactic', 'acetone', 'palmitate']

rows = []
for key, pw in ROUTE.items():
    r = P.loc[pw]
    assert r.q < 0.05 and r.q_nsti_adj < 0.05, f'{pw} does not survive both models'
    assert (r.beta > 0) == (key in POS), f'{pw} has the wrong sign for the map'
    rows += [(f'route_{key}_beta', r.beta), (f'route_{key}_q', r.q_nsti_adj)]

surv = P[(P.q < 0.05) & (P.q_nsti_adj < 0.05)]
rows += [('route_n_survive', len(surv)),
         ('route_n_survive_pos', int((surv.beta > 0).sum())),
         ('route_n_survive_neg', int((surv.beta < 0).sum())),
         ('route_n_nodes', len(POS))]

NEW = pd.DataFrame(rows, columns=['key', 'value'])
dup = set(NEW.key) & set(N.key)
assert not dup, f'key collision with the v58 numbers: {dup}'
out = pd.concat([N, NEW], ignore_index=True)
out.to_csv('numbers_20260912_v59.csv', index=False)
print(f'numbers_20260912_v59.csv: {len(N)} carried forward + {len(NEW)} new = {len(out)}')
print(f'  {len(surv)} pathways survive both models: {int((surv.beta>0).sum())} positive, '
      f'{int((surv.beta<0).sum())} negative')
