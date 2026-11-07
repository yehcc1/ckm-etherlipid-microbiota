"""Readable, unique labels for the SILVA taxonomy strings — shared by every v54 script.

Columns are Domain;Phylum;Class;Order;Family;Genus, with 'unclassified' filling any rank
that could not be assigned. A bin whose genus is unclassified is named for the deepest rank
that was assigned, so 'unclassified Erysipelotrichaceae' rather than 'unclassified'.
"""
RANK = ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
VOID = {'', 'nan', 'unclassified', 'uncultured', 'unknown'}


def label(full):
    parts = [p.strip() for p in full.split(';')]
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() not in VOID:
            return parts[i] if i == len(parts) - 1 else f'unclassified {parts[i]}'
    return full


def labels(columns):
    """labels for a whole column index, made unique by suffixing the parent rank."""
    out, seen = [], {}
    base = [label(c) for c in columns]
    for full, nm in zip(columns, base):
        if base.count(nm) > 1:
            parts = [p.strip() for p in full.split(';')]
            for p in reversed(parts[:-1]):
                if p.lower() not in VOID and p not in nm:
                    nm = f'{nm} ({p})'; break
        while nm in seen:
            nm += "'"
        seen[nm] = 1; out.append(nm)
    return out
