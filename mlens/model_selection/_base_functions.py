"""ML-Ensemble

Support functions for model selection suite.
"""
from __future__ import division

from ..parallel.learner import EvalLearner, EvalTransformer
from ..externals.sklearn.base import clone


def parse_key(key):
    """Helper to format keys"""
    key = key.split('__')
    case_est, draw = '__'.join(key[:-1]), key[-1]
    return case_est, draw


def check_scorer(scorer):
    """Check that the scorer instance passed behaves as expected."""
    if not type(scorer).__name__ in ['_PredictScorer', '_ProbaScorer']:

        raise ValueError(
            "The passes scorer does not seem to be a valid scorer. Expected "
            "type '_PredictScorer', got '%s'. Use the "
            "mlens.metrics.make_scorer function to construct a valid scorer." %
            type(scorer).__name__)


def cat(pr_name, est_name, union='__'):
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


def make_tansformers(generator, indexer, **kwargs):
    """Set up generators for the job to be performed"""
    transformers = [
        EvalTransformer(estimator=transformers, name=preprocess_name,
                        indexer=indexer, **kwargs)
        for preprocess_name, transformers in generator]
    return transformers


def make_learners(generator, indexer, scorer, error_score, **kwargs):
    """Set up generators for the job to be performed"""
    learners = [
        EvalLearner(
            estimator=clone(est).set_params(**params),
            preprocess=p_name, indexer=indexer,
            name='%s__%s' % (l_name, i) if i is not None else l_name,
            attr='predict', scorer=scorer, error_score=error_score, **kwargs)
        for p_name, l_name, est, i, params in generator]
    return learners

