import numpy as np

def code_mix_index(tags, langs):
    if len(np.unique(langs)) != len(langs):
        raise Exception('Languages provided should be unique.')

    counts = []
    for lang in langs:
        counts.append(tags.count(lang))

    w = np.max(counts)
    n = len(tags)
    u = n - np.sum(counts)

    return float(1 - (w/(n-u)) if n != u else 0)

#def switch_point_frac():
