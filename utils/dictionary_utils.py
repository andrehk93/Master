import numpy as np


# Merge dictionaries a into b
def merge_dicts(a, b):
    for key in b.keys():
        b[key].append(a[key])


# Creates a separate dictionary c, concatenated from a and b
def concatenate_dictionaries(a, b, b_from=0, a_from=0):
    c = {}
    for key in a.keys():
        c[key] = np.concatenate((a[key][a_from:], b[key][b_from:]))
        assert len(c[key]) == len(a[key][a_from:]) + len(b[key][b_from:]),\
            "Something went wrong during concatenation of dictionaries!"
    return c
