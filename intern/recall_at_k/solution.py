from typing import List
import numpy as np
from scipy.stats import hmean


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """
    Calculate the recall@k metric.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    _, labels = zip(*pairs)
    metric = np.sum(labels[:k]) / np.sum(labels)
    return metric


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """
    Calculate the precision@k metric.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    _, labels = zip(*pairs)
    metric = np.sum(labels[:k]) / k
    return metric


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """Calculate the specificity@k metric."""

    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    _, labels = zip(*pairs)
    tn = sum(1 - np.array(labels[k:]))
    fp = sum(1 - np.array(labels[:k]))
    if all([tn == 0, fp == 0]):
        metric = 0
        return metric
    metric = tn / (tn + fp)
    return metric


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """Calculate the f1@k metric."""
    metric = hmean([
        precision_at_k(labels=labels, scores=scores, k=k),
        recall_at_k(labels=labels, scores=scores, k=k)
    ])
    return metric
