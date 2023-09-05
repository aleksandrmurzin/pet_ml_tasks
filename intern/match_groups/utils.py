from typing import List
from typing import Tuple
from collections import defaultdict
import itertools


def extend_matches(pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    d = defaultdict(list)
    pairs = [tuple(sorted(i)) for i in pairs]
    for i in pairs:
        el1, el2 = i[0], i[1]
        d[el1].append(el2)
        d[el2].append(el1)

    for i in d:
        for y in d[i]:
            if y in d:
                d[i] += d[y]
                d[i] = list(set(d[i]))

    for i in d:
        if len(d[i]) >= 2:
            [pairs.append(p) for p in itertools.combinations(d[i], 2)
             if (p[0],p[1]) not in pairs and (p[1],p[0]) not in pairs]
    return sorted(pairs)
