"""Baseline for Kaggle AB."""

from typing import Callable, Dict, List, Union, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from tqdm import tqdm


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, Tuple[int, int]],
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 42,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Perform cross-validation and evaluate the performance of multiple model configurations.

    Parameters:
    model : callable
        The model to train and evaluate (e.g. RandomForestRegressor).
    X : numpy.ndarray
        The feature matrix.
    y : numpy.ndarray
        The target array.
    cv : int or tuple
        Number of folds for cross-validation or tuple (n_splits, n_repeats) for repeated K-Fold.
    params_list : list of dict
        A list of dictionaries, each representing a set of model parameters to be tested.
    scoring : callable
        Scoring function to evaluate model performance (e.g. r2_score).
    random_state : int, optional
        Random state for cross-validation. (Default is 42)
    show_progress : bool, optional
        If True, display a progress bar for the cross-validation process. (Default is False)

    Returns:
    numpy.ndarray
        Cross-validation scores for different models and folds. Shape: [n_models x n_folds].
    """

    if isinstance(cv, tuple):
        n_splits, n_repeats = cv
        kf = RepeatedKFold(n_splits=n_splits,
                           n_repeats=n_repeats, random_state=random_state)
        metrics = np.empty((0,  n_splits * n_repeats))

    if isinstance(cv, int):
        n_splits = cv
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        metrics = np.empty((0, n_splits))

    if show_progress:
        params_list = tqdm(params_list)

    for param in params_list:
        model.set_params(**param)
        fold_metrics = []
        for _, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, np.log1p(y_train))
            y_pred = np.expm1(model.predict(X_test))
            fold_metrics.append(scoring(y_test, y_pred))
        metrics = np.concatenate(
            (metrics, np.array(fold_metrics).reshape(1, -1)), axis=0)
    return metrics


def compare_models(
    cv: Union[int, Tuple[int, int]],
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    alpha: float = 0.05,
    show_progress: bool = False,
) -> List[Dict]:
    """
    Compare models using cross-validation and statistical tests.

    Parameters:
    cv : int or tuple
        Number of folds for cross-validation or tuple (n_splits, n_repeats) for repeated K-Fold.
    model : callable
        The model to train and evaluate (e.g. RandomForestRegressor).
    params_list : list of dict
        A list of dictionaries, each representing a set of model parameters to be tested.
    X : numpy.ndarray
        The feature matrix.
    y : numpy.ndarray
        The target array.
    random_state : int, optional
        Random state for cross-validation. (Default is 42)
    alpha : float, optional
        Significance level for statistical tests. The alpha value determines the threshold for
        accepting/rejecting the null hypothesis during the statistical comparison. (Default is 0.05)
    show_progress : bool, optional
        If True, display a progress bar for the cross-validation process. (Default is False)

    Returns:
    list of dict
        A list of dictionaries with model comparison results for different parameter sets.
        Each dictionary contains the following keys:

        - 'model_index': The index of the model in the params_list (starts from 1).
        - 'avg_score': The average cross-validation score for the model.
        - 'p_value': The p-value from the statistical test comparing the model to the baseline.
        It represents the probability of observing the given score difference between the model and the baseline due to random chance.
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

    results = []

    baseline = metrics[0, :]
    for idx, elem in enumerate(metrics[1:, :]):
        current_metric = elem.mean()
        statistics, p_value = ttest_rel(baseline, elem)
        if p_value < alpha and statistics > 0:
            effect_sign = -1
        elif p_value < alpha and statistics < 0:
            effect_sign = 1
        else:
            effect_sign = 0
        results.append({"model_index": idx+1,
                        "avg_score": current_metric,
                        "p_value": p_value,
                        "effect_sign": effect_sign})
    results = sorted(results, key=lambda x: (
        x["effect_sign"], x["avg_score"]), reverse=True)

    return results
