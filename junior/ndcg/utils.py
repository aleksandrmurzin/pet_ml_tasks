from typing import List
import numpy as np


def cumulative_gain(relevance: List[float], k: int) -> float:
    """_summary_

    Args:
        relevance (List[float]): _description_
        k (int): _description_

    Returns:
        float: _description_
    """
    return np.sum(relevance[:k])


def discounted_cumulative_gain(relevance, k, method):
    """_summary_

    Args:
        relevance (_type_): _description_
        k (_type_): _description_
        method (_type_): _description_

    Returns:
        _type_: _description_
    """
    denominator = list(np.log2(i + 1) for i in range(1, k+1))

    if method == "standard":
        numerator = relevance
    elif method == "industry":
        numerator = list(map(lambda x: 2**x - 1, relevance))
    else:
        raise ValueError

    dsg = np.sum(n / d for n, d in zip(numerator[:k], denominator[:k]))

    return dsg


def normalized_dcg(relevance, k, method='standard'):
    """_summary_

    Args:
        relevance (_type_): _description_
        k (_type_): _description_
        method (_type_): _description_

    Returns:
        _type_: _description_
    """
    score = (
        discounted_cumulative_gain(relevance, k, method)
        / discounted_cumulative_gain(
            sorted(relevance, reverse=True), k, method))
    return score


def avg_ndcg(list_relevances, k, method="standard"):
    """_summary_

    Args:
        list_relevances (_type_): _description_
        k (_type_): _description_
        method (str, optional): _description_. Defaults to "standard".

    Returns:
        _type_: _description_
    """
    return np.mean(list(normalized_dcg(r, k, method) for r in list_relevances))


# def test_cumulative_gain():
#     assert 4.26 == cumulative_gain(
#         relevance=[0.99, 0.94, 0.88, 0.74, 0.71, 0.68], k=5)


# def test_cumulative_gain():
#     assert 2.6164 == discounted_cumulative_gain(
#         relevance=[0.99, 0.94, 0.88, 0.74, 0.71, 0.68], k=5, method='standard')
