from typing import List
from typing import Tuple
from collections import defaultdict


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """_summary_

    Parameters
    ----------
    pairs : List[Tuple[int, int]]
        _description_

    Returns
    -------
    List[Tuple[int, int]]
        _description_
    """
    d = defaultdict(list)
    pairs = [tuple(sorted(i)) for i in pairs]
    for i in pairs:
        for el in i:
            d[el] += [e for e in i if e != el]
    for i in d:
        for y in d[i]:
            if y in d:
                d[i] += d[y]
                d[i] = list(set(d[i]))
        if i not in d[i]:
            d[i].append(i)

    pairs = [tuple(sorted(v)) for _, v in d.items()]
    return sorted(list(set(pairs)))