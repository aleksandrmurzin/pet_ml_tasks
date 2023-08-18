from typing import List, Tuple

import numpy as np
from scipy.stats import ttest_ind


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    _, p_value = ttest_ind(control, experiment)
    result = bool(p_value < alpha)
    return p_value, result



def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    def bootstrap(control, experiment, quantile, n_bootstraps):
        """_summary_

        Parameters
        ----------
        control : _type_
            _description_
        experiment : _type_
            _description_
        quantile : _type_
            _description_
        n_bootstraps : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        control_q, experiment_q = [], []
        for _ in range(n_bootstraps):
            a_strapped = np.random.choice(
                control, size=len(control), replace=True)
            b_strapped = np.random.choice(
                experiment, size=len(experiment), replace=True)

            control_q.append(np.quantile(a_strapped, quantile))
            experiment_q.append(np.quantile(b_strapped, quantile))
        return control_q, experiment_q
    p_value, result = ttest(
        *bootstrap(control, experiment, quantile, n_bootstraps), alpha=alpha)
    return p_value, result
