"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Class for parallel tuning a set of estimators that share a common
preprocessing pipeline.
"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments

from __future__ import division, with_statement

import warnings
import numpy as np

from .. import config
from ..index import FoldIndex
from ..parallel import ParallelEvaluation
from ..parallel.learner import EvalLearner, Transformer
from ..metrics import Data, assemble_data
from ..utils import (print_time,
                     safe_print,
                     check_instances,
                     assert_correct_format,
                     check_inputs)
from ..externals.joblib import delayed
from ..externals.sklearn.base import clone

try:
    from time import perf_counter as time
except ImportError:
    from time import time

try:
    from collections import OrderedDict as _dict
except ImportError:
    _dict = dict


def parse_key(key):
    """Helper to format keys"""
    draw = key[-1]
    case_est = key[:-1]
    if len(case_est) == 1:
        case_est = case_est[0]
    return case_est, draw


def _check_scorer(scorer):
    """Check that the scorer instance passed behaves as expected."""
    if not type(scorer).__name__ in ['_PredictScorer', '_ProbaScorer']:

        raise ValueError("The passes scorer does not seem to be a valid "
                         "scorer. Expected type '_PredictScorer', got '%s'."
                         "Use the mlens.metrics.make_scorer function to "
                         "construct a valid scorer." % type(scorer).__name__)


def _name(case, est_name):
    """Get correct param_dict name."""
    if case is not None:
        return case.split('__')[0], est_name.split('__')[0]
    return est_name.split('__')[0]


class Evaluator(object):

    r"""Model selection across several estimators and preprocessing pipelines.

    The :class:`Evaluator` allows users to evaluate several models in one call
    across a set preprocessing pipelines. The class is useful for comparing
    a set of estimators, especially when several preprocessing pipelines is to
    be evaluated. By pre-making all folds and iteratively fitting estimators
    with different parameter settings, array slicing and preprocessing is kept
    to a minimum. This can greatly reduced fit time compared to
    creating pipeline classes for each estimator and pipeline and fitting them
    one at a time in an Scikit-learn
    :class:`sklearn.model_selection.GridSearch` class.

    Preprocessing can be done before making any evaluation, and several
    evaluations can be made on the pre-made folds. Current implementation
    relies on a randomized grid search, so parameter grids must be specified as
    SciPy distributions (or a class that accepts a ``rvs`` method).

    Parameters
    ----------
    scorer : function
        a scoring function that follows the Scikit-learn API::

            score = scorer(estimator, y_true, y_pred)

        A user defines scoring function, ``score = f(y_true, y_pred)`` can be
        made into a scorer by calling on the ML-Ensemble implementation of
        Scikit-learn's ``make_scorer``. NOTE: do **not** use Scikit-learn's
        ``make_scorer`` if the Evaluator is to be pickled. ::

            from mlens.metrics import make_scorer
            scorer = make_scorer(scoring_function, **kwargs)

    error_score : int, optional
        score to assign when fitting an estimator fails. If ``None``, the
        evaluator will raise an error.

    cv : int or obj (default = 2)
        cross validation folds to use. Either pass a ``KFold`` class
        that obeys the Scikit-learn API.

    metrics : list, optional
        list of aggregation metrics to calculate on scores. Default is
        mean and standard deviation.

    shuffle : bool (default = True)
        whether to shuffle input data before creating cv folds.

    random_state : int, optional
        seed for creating folds (if shuffled) and parameter draws

    array_check : int (default = 2)
        level of strictness in checking input arrays.

            - ``array_check = 0`` will not check ``X`` or ``y``

            - ``array_check = 1`` will check ``X`` and ``y`` for
              inconsistencies and warn when format looks suspicious,
              but retain original format.

            - ``array_check = 2`` will impose Scikit-learn array checks,
              which converts ``X`` and ``y`` to numpy arrays and raises
              an error if conversion fails.

    n_jobs: int (default = -1)
        number of CPU cores to use.

    verbose : bool or int (default = False)
        level of printed messages.

    Attributes
    ----------
    summary : dict
        Summary output that shows data for best mean test scores, such as
        test and train scores, std, fit times, and params.

    cv_results : dict
        a nested ``dict`` of data from each fit. Includes mean and std of
        test and  train scores and fit times, as well as param draw index
        and parameters.
    """

    def __init__(self,
                 scorer,
                 cv=2,
                 shuffle=True,
                 random_state=None,
                 backend=None,
                 error_score=None,
                 metrics=None,
                 array_check=2,
                 n_jobs=-1,
                 verbose=False):

        self.cv = cv
        self.indexer = FoldIndex(cv)
        self.shuffle = shuffle
        self.backend = backend if backend is not None else config.BACKEND
        self.n_jobs = n_jobs
        self.error_score = error_score
        self.metrics = [np.mean, np.std] if metrics is None else metrics
        self.array_check = array_check
        self.random_state = random_state
        self.verbose = verbose
        self._preprocessing = None
        self._transformers = None
        self._estimators = None
        self._learners = None
        self.n_iter = None
        self.params = None
        self.results = None

        _check_scorer(scorer)
        self.scorer = scorer
        self.scores_ = None

    def __iter__(self):
        """Provide jobs for ParallelEvaluation manager"""
        yield self

    def __call__(self, parallel, args, case):
        """Process eval"""
        job = args['job']
        path = args['dir']

        if self.verbose:
            printout = "stderr" if self.verbose < 50 else "stdout"
            safe_print('Launching job', file=printout)
            t0 = time()

        if ('preprocess' in case) or (self._transformers):
            # Second test is for already fitted pipes - need to be cached
            if self.verbose >= 2:
                printout = "stderr" if self.verbose < 50 else "stdout"
                safe_print('Preparing preprocess pipelines', file=printout)
                t1 = time()

            parallel(delayed(subtransformer)(job, path)
                     for transformer in self._transformers
                     for subtransformer
                     in transformer(job, **args['transformer']))

            if 'preprocess' in case:
                self.collect(args['dir'], 'transformers')

            if self.verbose >= 2:
                print_time(t1, 'Done', file=printout)

        if 'evaluate' in case:
            if self.verbose >= 2:
                printout = "stderr" if self.verbose < 50 else "stdout"
                safe_print('Evaluating estimators', file=printout)
                t1 = time()

            parallel(delayed(sublearner)(job, path)
                     for learner in self._learners
                     for sublearner in learner(job, **args['learner']))

            self.collect(args['dir'], 'estimators')
            if self.verbose >= 2:
                print_time(t1, 'Done', file=printout)

        if self.verbose:
            print_time(t0, 'Done', file=printout)

    def collect(self, path, case):
        """Collect cache estimators"""
        if case == 'transformers':
            for transformer in self._transformers:
                transformer.collect(path)
        if case == 'estimators':
            for learner in self._learners:
                learner.collect(path)
            self._get_results()

    def get_raw_data(self):
        """Cross validated scores"""
        data = list()
        for learner in self._learners:
            data.extend(learner.raw_data)
        return assemble_data(data)

    def fit(self, X, y,
            estimators=None,
            param_dicts=None,
            n_iter=2,
            preprocessing=None):
        """Fit the Evaluator to given data, estimators and preprocessing.

        Utility function that calls ``preprocess`` and ``evaluate``. The
        following is equivalent::

            # Explicitly calling preprocess and evaluate
            evaluator.preprocess(X, y, preprocessing)
            evaluator.evaluate(X, y, estimators, param_dicts, n_iter)

            # Calling fit
            evaluator.fit(X, y, estimators, param_dicts, n_iter, preprocessing)

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input data to preprocess and create folds from.

        y : array-like, shape=[n_samples, ]
            training labels.

        estimators : list or dict
            set of estimators to use. If no preprocessing is desired or if
            only on preprocessing pipeline should apply to all, pass a list of
            estimators. The list can contain elements of named tuples
            (i.e. ``('my_name', my_est)``).

            If different estimators should be mapped to preprocessing cases,
            a dictionary that maps estimators to each case should
            be passed: ``{'case_a': list_of_est, ...}``.

        param_dicts : dict
            parameter distribution mapping for estimators. Current
            implementation only supports randomized grid search. Passed
            distribution object must have an ``rvs`` method.
            See :mod:`Scipy.stats` for details.

            There is quite some flexibility in specifying ``param_dicts``. If
            there is no preprocessing, or if all estimators are fitted on all
            preprocessing cases, the ``param_dict`` should have keys matching
            the names of the estimators. ::

                estimators = [('name', est), est]

                param_dicts = {'name': {'param-1': some_distribution},
                               'est': {'param-1': some_distribution}
                              }

            It is possible to specify different distributions for some or all
            preprocessing cases::

                preprocessing = {'case-1': transformer_list,
                                 'case-2': transformer_list}

                estimators = [('name', est), est]

                param_dicts = {'name':
                                   {'param-1': some_distribution},
                               ('case-1', 'est'):
                                   {'param-1': some_distribution}
                               ('case-2', 'est'):
                                   {'param-1': some_distribution,
                                    'param-2': some_distribution}
                              }

            If estimators are mapped on a per-preprocessing case basis as a
            dictionary, ``param_dict`` must have key entries of the form
            ``(case_name, est_name)``.

        n_iter : int
            number of parameter draws to evaluate.

        preprocessing : dict, optional
            preprocessing cases to consider. Pass a dictionary mapping a
            case name to a preprocessing pipeline. ::

                preprocessing = {'case_name': transformer_list,}

        Returns
        -------
        self : instance
            class instance with stored estimator evaluation results.
        """
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
        self._initialize(job, estimators, preprocessing, param_dicts, n_iter)

        X, y = check_inputs(X, y, self.array_check)

        with ParallelEvaluation(self) as manager:
            manager.process(job, X, y)

        return self

    def _initialize(self, job, estimators, preprocessing, param_dicts, n_iter):
        """Set up generators for the job to be performed"""
        if 'preprocess' in job:
            self._preprocessing = check_instances(preprocessing)

            self._transformers = [
                Transformer(pipeline=transformers,
                            name=preprocess_name,
                            indexer=self.indexer,
                            raise_on_exception=True)
                for preprocess_name, transformers
                in sorted(self._preprocessing.items())
            ]

        if 'evaluate' in job:
            estimators, param_dicts = self._format(estimators, param_dicts)
            self._estimators, flattened_estimators = check_instances(
                estimators, include_flattened=True)
            self.n_iter = n_iter
            self._param_sets(param_dicts)

            self._learners = [
                EvalLearner(estimator=clone(est).set_params(**params),
                            preprocess=preprocess_name,
                            indexer=self.indexer,
                            name='%s__%s' % (learner_name, i),
                            attr='predict',
                            scorer=self.scorer,
                            error_score=self.error_score,
                            raise_on_exception=True)
                for preprocess_name, learner_name, est in flattened_estimators
                for i, params in enumerate(
                    self.params[_name(preprocess_name, learner_name)])
            ]

    def _format(self, estimators, param_dicts):
        """Ensure estimator object and param_dict object have right format."""
        preprocessing = self._preprocessing

        if preprocessing is not None and isinstance(estimators, list):
            # Set all estimators in list as ests for each case
            ests_ = estimators
            estimators = {case: ests_ for case in preprocessing.keys()}

        # Set parameter draws for each case
        if preprocessing is not None:

            params = dict()
            for key, pars in param_dicts.items():
                if isinstance(key, tuple):
                    # Check that naming is of the (case, est) form

                    if key[0] not in preprocessing:
                        msg = ("param_dict poorly formatted. Valid keys are "
                               "'(case_name, est_name)' or 'est_name'."
                               " Failed on key entry {}. \nAll keys: "
                               "{}".format(key, list(preprocessing)))

                        raise ValueError(msg)

                    params[key] = pars

                else:
                    # have an est_name key entry. Need to generate
                    # keys of the form (case, est)
                    for case in preprocessing.keys():
                        if (case, key) in params:
                            # We do not want to overwrite user-specified dists
                            continue

                        params[(case, key)] = pars
        else:
            params = param_dicts

        # Finally, check that estimators and preprocessing are correctly
        # formatted for estimation
        assert_correct_format(estimators, preprocessing)

        return estimators, params

    def _draw_params(self, param_dists):
        """Draw a list of param dictionaries for estimator."""
        # Set up empty list of parameter setting
        param_draws = [{} for _ in range(self.n_iter)]

        # Fill list of parameter settings by param
        for param, dist in param_dists.items():
            draws = dist.rvs(size=self.n_iter, random_state=self.random_state)

            for i, draw in enumerate(draws):
                param_draws[i][param] = draw

        return param_draws

    def _set_params(self, param_dicts, key):
        """Try to set params, and if failure set an empty list."""
        try:
            self.params[key] = \
                self._draw_params(param_dicts[key])
        except KeyError:
            # No param draws desired. Set empty dict.
            warnings.warn("No valid parameters found for {}. Will fit and "
                          "score once with given parameter "
                          "settings.".format(key))
            self.params[key] = [{}]

    def _param_sets(self, param_dicts):
        """For each estimator, create a mapping of parameter draws."""
        self.params = dict()

        if not self._preprocessing:
            # No preprocessing
            # the expected param_dicts key is 'est_name'
            for est_name, _ in self._estimators:
                self._set_params(param_dicts, est_name)
        else:
            # Preprocessing
            # Iterate over cases, expected param_dicts key is
            # ('case_name', 'est_name')
            if isinstance(self._preprocessing, dict):
                for case in self._preprocessing:
                    for est_name, _ in self._estimators[case]:
                        self._set_params(param_dicts, (case, est_name))
            else:
                for est_name, _ in self._estimators:
                    self._set_params(param_dicts, (None, est_name))

    def _get_results(self):
        """For each case-estimator, return best param draw from cv results."""
        data = self.get_raw_data()
        best = _dict()
        for key, val in data.items():
            best[key] = _dict()
            for k in val.keys():
                case_est, _ = parse_key(k)
                best[key][case_est] = None

        best['params'] = _dict()
        for k in data['test_score-m'].keys():
            case_est, _ = parse_key(k)
            best['params'][case_est] = None

        for key, score in data['test_score-m'].items():
            case_est, draw = parse_key(key)

            old_score = best['test_score-m'][case_est]
            if old_score is None or score > old_score:
                best['test_score-m'][case_est] = score
                for k, val in data.items():
                    best[k][case_est] = val[key]
                best['params'][case_est] = self.params[case_est][int(draw)]

        self.results = Data(best)

    def _print_prep_start(self, t0, printout):
        """Print preprocessing start and return timer."""
        msg = 'Preprocessing %i preprocessing pipelines over %i CV folds'

        p = len(getattr(self, 'preprocessing', [1]))
        c = self.cv if isinstance(self.cv, int) else self.cv.n_splits
        safe_print(msg % (p, c), file=printout)
        return t0

    def _print_eval_start(self, printout):
        """Print initiation message and return timer."""
        preprocessing = getattr(self, 'preprocessing', None)

        if preprocessing is None:

            msg = ('Evaluating %i models for %i parameter draws over %i '
                   'CV folds, totalling %i fits')

            e, c, tot = self._get_count(preprocessing)
            safe_print(msg % (e, self.n_iter, c, tot), file=printout)
        else:

            msg = ('Evaluating %i models for %i parameter draws over %i' +
                   ' preprocessing pipelines and %i CV folds, '
                   'totalling %i fits')

            e, p, c, tot = self._get_count(preprocessing)
            safe_print(msg % (e, self.n_iter, p, c, tot), file=printout)

    def _get_count(self, preprocessing):
        """Utility for counting number of fits to make."""
        c = self.cv

        if preprocessing is None:
            # Simply grab length of estimator list
            e = len(self._estimators)
            tot = e * c * self.n_iter
            return int(e), int(c), int(tot)
        else:
            # Need to consider cases
            p = len(preprocessing)

            if isinstance(self._estimators, list):
                # If all estimators are applied to all cases, just grab
                # length of list and multiply by cases
                e = len(self._estimators) * p
            else:
                # Sum over cases
                e = 0
                for v in self._estimators.values():
                    e += len(v)

            tot = e * c * self.n_iter
            return int(e), int(p), int(c), int(tot)
