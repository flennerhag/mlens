"""ML-ENSEMBLE

author: Sebastian Flennerhag
licence:MIT
Functions for parallel preprocessing
"""

from __future__ import division, print_function

from ..base import safe_slice


def _preprocess_pipe(xtrain, ytrain, xtest, steps, fit, p_name=None,
                     dry_run=False, return_estimator=False):
    """Function to fit and transform all data with preprocessing pipeline."""
    for step in steps:
        if fit:
            step.fit(xtrain, ytrain)
        xtrain = step.transform(xtrain)
        if xtest is not None:
            xtest = step.transform(xtest)

    if dry_run:
        return
    else:
        if return_estimator:
            out = [steps, xtrain]
        else:
            out = [xtrain]
        if xtest is not None:
            out.append(xtest)
        if p_name is not None:
            out.append(p_name)

        return out


def _preprocess_fold(X, y, indices, preprocessing, fit=True, return_idx=True):
    """Function to fit and transform a fold with a preprocessing pipeline."""
    train_idx, test_idx = indices
    xtrain = safe_slice(X, row_slice=train_idx)
    xtest = safe_slice(X, row_slice=test_idx)

    try:
        ytrain = safe_slice(y, row_slice=train_idx)
        ytest = safe_slice(y, row_slice=test_idx)
    except Exception:
        ytrain, ytest = None, None

    if preprocessing is not None:
        p_name, preprocess_case = preprocessing
        out = _preprocess_pipe(xtrain, ytrain, xtest, preprocess_case, fit)
    else:
        p_name = ''
        out = [xtrain, xtest]

    out += [ytrain, ytest]
    if return_idx:
        out.append(test_idx)

    out.append(p_name)

    return out
