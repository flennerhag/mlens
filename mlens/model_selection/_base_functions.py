"""ML-Ensemble

Support functions for model selection suite.
"""
from __future__ import division

from collections import Counter
from ..parallel import EvalLearner, EvalTransformer, Pipeline
from ..utils.formatting import _assert_format
from ..utils.checks import assert_valid_estimator
from ..externals.sklearn.base import clone


def parse_key(key):
    """Helper to format keys"""
    key = key.split('.')
    case_est, draw = '.'.join(key[:-1]), key[-1]
    return case_est, draw


def check_scorer(scorer):
    """Check that the scorer instance passed behaves as expected."""
    if not type(scorer).__name__ in ['_PredictScorer', '_ProbaScorer']:

        raise ValueError(
            "The passes scorer does not seem to be a valid scorer. Expected "
            "type '_PredictScorer', got '%s'. Use the "
            "mlens.metrics.make_scorer function to construct a valid scorer." %
            type(scorer).__name__)


def cat(pr_name, est_name, union='.'):
    """Concat preprocess and estimator name if applicable."""
    if not pr_name:
        return est_name
    return union.join([pr_name, est_name])


def set_job(estimators, preprocessing):
    """Set job to run"""
    if estimators is None:
        if preprocessing is None:
            raise ValueError("Need to specify at least one of"
                             "[estimators, preprocessing]")
        else:
            job = 'preprocess'
    elif preprocessing is None:
        job = 'evaluate'
    else:
        job = 'preprocess-evaluate'
    return job


def _format_instances(instances):
    """Format a list of instances to a list of named estimator tuples."""
    named_instances = []
    for val in instances:
        # Check that the instance appears correctly specified
        if not isinstance(val, (list, tuple, set)):
            # val is the instance
            instance = val
        else:
            # val is a list-like object. Assume instance is the last entry
            instance = val[-1]

        # Check if it appears to be an estimator
        assert_valid_estimator(instance)

        try:
            # Format instance names

            # We keep the instance as a list to change possible duplicate names
            # exploiting that lists are mutable, before switching to tuple
            if instance == val:
                tup = [instance.__class__.__name__.lower(), instance]
            else:
                tup = ['-'.join(val[0].split()).lower(), val[-1]]

            named_instances.append(tup)

        except Exception as e:
            msg = ("Could not format instance %s. Check that passed instance "
                   "iterables follow correct syntax:\n"
                   "- if multiple preprocessing cases, pass a dictionary with "
                   "instance lists as values and case name as key.\n"
                   "- else, pass list of (named) instances.\n"
                   "See documentation for further information.\n"
                   "Error details: %r")
            raise ValueError(msg % (instance, e))

    # Check and correct duplicate names
    duplicates = Counter([tup[0] for tup in named_instances])
    duplicates = {key: val for key, val in duplicates.items() if
                  val > 1}

    out = []  # final named_instances list

    name_count = {key: 1 for key in duplicates}
    for name, instance in named_instances:
        if name in duplicates:
            current_name_count = name_count[name]  # fix before update
            name_count[name] += 1
            name += '-%d' % current_name_count  # rename
        out.append((name, instance))
    return sorted(out)


def check_instances(instances):
    """Helper to ensure all instances are named.

    Check if ``instances`` is formatted as expected, and if not convert
    formatting or throw traceback error if impossible to anticipate formatting.

    Parameters
    ----------
    instances : iterable
        instance iterable to test.

    Returns
    -------
    formatted : list or dict
        formatted ``instances`` object. Will be formatted as a dict if
        preprocessing cases are detected, otherwise as a list. The dict will
        contain lists identical to those in the single preprocessing case.
        Each list is of the form ``[('name', instance)]`` and no names overlap.

    Raises
    ------
    ValueError:
        Raises error if formatting fails, which is most likely due to wrong
        ordering of tuple entries, or wrong argument in the wrong position.
    """
    is_iterable = isinstance(instances, (list, tuple, dict))
    if not is_iterable:
        instances = [instances]
    if not instances or None in instances:
        return None

    if _assert_format(instances):
        out = instances
    elif isinstance(instances, list):
        out = _format_instances(instances)
    else:
        # We need to check the instance list of each case
        out = {}
        for case, case_list in instances.items():
            out['-'.join(case.lower().split())] = \
                _format_instances(case_list)
    return out


def make_tansformers(generator, indexer, **kwargs):
    """Set up generators for the job to be performed"""
    transformers = [
        EvalTransformer(estimator=Pipeline(pipeline, return_y=True),
                        name=preprocess_name, indexer=indexer, **kwargs)
        for preprocess_name, pipeline in generator]
    return transformers


def make_learners(generator, indexer, scorer, error_score, **kwargs):
    """Set up generators for the job to be performed"""
    learners = [
        EvalLearner(
            estimator=clone(est).set_params(**params),
            preprocess=p_name, indexer=indexer,
            name='%s.%s' % (l_name, i) if i is not None else l_name,
            attr='predict', scorer=scorer, error_score=error_score, **kwargs)
        for p_name, l_name, est, i, params in generator]
    return learners
