"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 42,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Cross-validation score.

    Parameters
    ----------
    model : callable
        Model to train (e.g. RandomForestRegressor).
    X : numpy.ndarray
        The feature matrix.
    y : numpy.ndarray
        The target array.
    cv : int
        Number of folds for cross-validation.
    params_list : list of dict
        List of model parameters to be tested. Each element is a dictionary representing a set of parameters.
    scoring : callable
        Scoring function to evaluate model performance (e.g. r2_score).
    random_state : int, optional
        Random state for cross-validation. (Default is 42)
    show_progress : bool, optional
        If True, display a progress bar for the cross-validation process. (Default is False)

    Returns
    -------
    numpy.ndarray
        Cross-validation scores for different models and folds. Shape: [n_models x n_folds].
     """
    metrics = np.empty((0, cv))
    if show_progress:
        params_list = tqdm(params_list)
    for param in params_list:
        model = model.set_params(**param)
        fold_metrics = []
        kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        for _, (train_index, test_index) in enumerate(kf.split(X)):
            y_pred = model.fit(X[train_index], y[train_index])\
                .predict(X[test_index])
            fold_metrics.append(scoring(y[test_index], y_pred))
        metrics = np.concatenate(
            (metrics, np.array(fold_metrics).reshape(1, -1)), axis=0)
    return metrics


def compare_models(
    cv: int,
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    show_progress: bool = False,
) -> List[Dict]:
    """
    Compare models using cross-validation.
    Parameters
    ----------
    cv : int
        Number of folds for cross-validation.
    model : callable
        Model to train (e.g. RandomForestRegressor).
    params_list : list of dict
        List of model parameters to be tested. Each element is a dictionary representing a set of parameters.
    X : numpy.ndarray
        The feature matrix.
    y : numpy.ndarray
        The target array.
    random_state : int, optional
        Random state for cross-validation. (Default is 42)
    show_progress : bool, optional
        If True, display a progress bar for the cross-validation process. (Default is False)
    Returns
    -------
    list of dict
        A list of dictionaries with model comparison results for different parameter sets.
        Each dictionary contains the following keys:
        - 'model_index': The index of the model in the params_list (starts from 1).
        - 'avg_score': The average cross-validation score for the model.
        - 'effect_sign': An integer representing the effect of the model compared to the baseline:
                         1 if the model's performance is better than the baseline,
                         -1 if the model's performance is worse than the baseline,
                         0 if the model's performance is equal to the baseline.
    """

    metrics = cross_val_score(
        model=model,
        X=X,
        y=y,
        cv=cv,
        params_list=params_list,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )
    base_metric = metrics[0, :].mean()

    results = []

    for idx, elem in enumerate(metrics[1:, :]):
        current_metric = elem.mean()
        if current_metric > base_metric:
            sign = 1
        elif current_metric < base_metric:
            sign = -1
        else:
            sign = 0
        results.append({"model_index": idx+1,
                        "avg_score": current_metric,
                        "effect_sign": sign})
    results = sorted(results, key=lambda x: x["avg_score"], reverse=True)
    return results
