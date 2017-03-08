"""ML-ENSEMBLE

author: Sebastian Flennerhag
licence: MIT
Support functions for cloning ensemble estimators
"""

from __future__ import division, print_function
from sklearn.base import clone


def clone_base_estimators(base_estimators, as_dict=True):
    """Created named clones of base estimators for fitting."""
    if isinstance(base_estimators, list):
        base_estimators = [('', base_estimators)]
    else:
        base_estimators = [(case, ests) for case, ests in
                           base_estimators.items()]

    if as_dict:
        return {case: [(est_name, clone(est)) for est_name, est in
                       estimators]
                for case, estimators in base_estimators}
    else:
        return [(case, [(est_name, clone(est)) for est_name, est in
                        estimators])
                for case, estimators in base_estimators]


def clone_preprocess_cases(preprocess):
    """Created named clones of base preprocessing pipes for fitting."""
    if preprocess is None:
        return

    if isinstance(preprocess, dict):
        return [(case, [clone(trans) for _, trans in process_list])
                for case, process_list in preprocess.items()]
    else:
        return [('', [clone(trans) for _, trans in preprocess])]
