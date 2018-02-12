"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Model selection suite for tuning and benchmarking a set of estimators.
"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments

from __future__ import division, with_statement

import warnings
import numpy as np

from ._base_functions import (parse_key, set_job, cat, check_scorer,
                              make_learners, make_tansformers, check_instances)
from ..index import FoldIndex
from ..parallel import ParallelEvaluation
from ..parallel.base import BaseBackend, IndexMixin
from ..metrics import Data, assemble_data
from ..utils.formatting import _flatten, _check_instances
from ..utils import (print_time, safe_print,
                     assert_correct_format, check_inputs)
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


def benchmark(X, y, scorer, cv, estimators,
              preprocessing, error_score=None, **kwargs):
    """Benchmark estimators across preprocessing pipelines.

    :func:`benchmark` runs cross validation scoring of a set of estimators,
    possible against a set of preprocessing pipelines. Equivalent to ::

            evl = Benchmark(**kwargs)
            evl.fit(X, y, scorer, ...)


    .. versionadded:: 0.2.0

    Parameters
    ----------
    X : array-like, shape=[n_samples, n_features]
       input data to preprocess and create folds from.

    y : array-like, shape=[n_samples, ]
       training labels.

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

    estimators : list or dict, optional
        set of estimators to use. If no preprocessing is desired or if
        only on preprocessing pipeline should apply to all, pass a list of
        estimators. The list can contain elements of named tuples
        (i.e. ``('my_name', my_est)``).

        If different estimators should be mapped to preprocessing cases,
        a dictionary that maps estimators to each case should
        be passed: ``{'case_a': list_of_est, ...}``.

    preprocessing : dict, optional
        preprocessing cases to consider. Pass a dictionary mapping a
        case name to a preprocessing pipeline. ::

            preprocessing = {'case_name': transformer_list,}

    **kwargs : optional
        Optional arguments to :class:`~mlens.parallel.base.BaseBackend`.

    Returns
    -------
    results : dict
        Summary output that shows data for best mean test scores, such as
        test and train scores, std, fit times, and params.
    """
    evl = Benchmark(**kwargs)
    evl.fit(X, y, scorer, cv, estimators, preprocessing, error_score)
    return evl.results


class BaseEval(IndexMixin, BaseBackend):

    """Base Evaluation class."""

    def __init__(self, verbose=False, array_check=2, **kwargs):
        self.verbose = verbose
        self.array_check = array_check
        self._transformers = None
        self._learners = None
        super(BaseEval, self).__init__(**kwargs)

    def __iter__(self):
        """Provide jobs for ParallelEvaluation manager"""
        yield self

    def __call__(self, parallel, args, case):
        """Process eval"""
        if self.verbose:
            f = "stdout" if self.verbose < 20 else "stderr"
            safe_print('Launching job', file=f)
            t0 = time()

        if 'preprocess' in case or self._transformers:
            # Second test is for already fitted pipes - need to be cached
            if self.verbose >= 2:
                safe_print(self._print_prep_start(), file=f)
                t1 = time()

            self._run('transformers', parallel, args)
            if 'preprocess' in case:
                self.collect(args['dir'], 'transformers')

            if self.verbose >= 2:
                print_time(t1, '{:<13} done'.format('Preprocessing'), file=f)

        if 'evaluate' in case:
            if self.verbose >= 2:
                safe_print(self._print_eval_start(), file=f)
                t1 = time()

            self._run('estimators', parallel, args)
            self.collect(args['dir'], 'estimators')

            if self.verbose >= 2:
                print_time(t1, '{:<13} done'.format('Evaluation'), file=f)

        if self.verbose:
            print_time(t0, '{:<13} done'.format('Job'), file=f)

    def _run(self, case, parallel, args):
        """Process eval"""
        path = args['dir']
        _threading = self.backend == 'threading'

        if case == 'transformers':
            generator = self._transformers
            inp = 'auxiliary'
        else:
            generator = self._learners
            inp = 'main'

        parallel(delayed(subtask, not _threading)()
                 for task in generator for subtask in task(args, inp))

    def _fit(self, X, y, job):
        X, y = check_inputs(X, y, self.array_check)
        verbose = max(self.verbose - 2, 0) if self.verbose < 15 else 0
        with ParallelEvaluation(self.backend, self.n_jobs, verbose) as manager:
            manager.process(self, job, X, y)

    def collect(self, path, case):
        """Collect cache estimators"""
        if case == 'transformers':
            for transformer in self._transformers:
                transformer.collect(path)
        if case == 'estimators':
            for learner in self._learners:
                learner.collect(path)

    @property
    def raw_data(self):
        """Cross validated scores"""
        data = list()
        for learner in self._learners:
            data.extend(learner.raw_data)
        return assemble_data(data)

    def _print_prep_start(self):
        """Message at start of preprocessing"""
        return "Preprocessing"

    def _print_eval_start(self):
        """Message at start of preprocessing"""
        return "Evaluating"


class Benchmark(BaseEval):

    """Benchmark engine without hyper-parameter grid search.

    A simplified version of the :class:`Evaluator` that performs a single
    pass over a set of estimators and preprocessing pipelines for
    benchmarking purposes.

    .. versionadded:: 0.2.0

    Parameters
    ----------
    verbose : bool, int, optional
        Verbosity during estimation.

    **kwargs : optional
        Optional keyword argument to :class:`~mlens.parallel.base.BaseBackend`.
    """

    def __init__(self, verbose=False, **kwargs):
        super(Benchmark, self).__init__(verbose=verbose, **kwargs)
        self.results = None
        self.indexer = None

    def fit(self, X, y, scorer, cv, estimators,
            preprocessing=None, error_score=None):
        """Run benchmarking job on given data with given estimators.

        Fit preprocessing if applicable and evaluate estimators if applicable.
        The method automatically determines whether to only run preprocessing,
        only evaluation (possibly on previously fitted preprocessing), or both.
        Calling ``fit`` will overwrite previously stored data where applicable.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
           input data to preprocess and create folds from.

        y : array-like, shape=[n_samples, ]
           training labels.

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

        estimators : list or dict, optional
            set of estimators to use. If no preprocessing is desired or if
            only on preprocessing pipeline should apply to all, pass a list of
            estimators. The list can contain elements of named tuples
            (i.e. ``('my_name', my_est)``).

            If different estimators should be mapped to preprocessing cases,
            a dictionary that maps estimators to each case should
            be passed: ``{'case_a': list_of_est, ...}``.

        preprocessing : dict, optional
            preprocessing cases to consider. Pass a dictionary mapping a
            case name to a preprocessing pipeline. ::

                preprocessing = {'case_name': transformer_list,}

        Returns
        -------
        self : inst
            Fitted Benchmark instance. Results available in the
            ``results`` attribute.
        """
        self.indexer = FoldIndex(folds=cv)
        assert_correct_format(estimators, preprocessing)
        if preprocessing is not None:
            self._transformers = make_tansformers(
                sorted(check_instances(preprocessing).items()), self.indexer,
                verbose=max(0, self.verbose - 14))

        generator = [
            (p_name, l_name, est, None, {})
            for p_name, l_name, est in _flatten(check_instances(estimators))]

        self._learners = make_learners(
            generator, self.indexer, scorer, error_score,
            verbose=max(0, self.verbose - 14))

        job = set_job(estimators, preprocessing)
        self._fit(X, y, job)
        self.results = Data(self.raw_data, decimals=3)
        return self


class Evaluator(BaseEval):

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
    SciPy distributions (or a class that accepts an ``rvs`` method).

    .. versionchanged:: 0.2.0

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

    cv : int or obj, default = 2
        cross validation folds to use. Either pass a ``KFold`` class
        that obeys the Scikit-learn API.

    metrics : list, optional
        list of aggregation metrics to calculate on scores. Default is
        mean and standard deviation.

    shuffle : bool, default = True
        whether to shuffle input data before creating cv folds.

    random_state : int, optional
        seed for creating folds (if shuffled) and parameter draws

    array_check : int, default = 2
        level of strictness in checking input arrays.

            - ``array_check = 0`` will not check ``X`` or ``y``

            - ``array_check = 1`` will check ``X`` and ``y`` for
              inconsistencies and warn when format looks suspicious,
              but retain original format.

            - ``array_check = 2`` will impose Scikit-learn array checks,
              which converts ``X`` and ``y`` to numpy arrays and raises
              an error if conversion fails.

    n_jobs: int, default = -1
        number of CPU cores to use.

    verbose : bool or int, default = False
        level of printed messages. Levels:

            #. ``verbose=1``: Message at start and end with total time
            #. ``verbose=2``: Additional messages for each sub-job \
               (preprocess and evaluation)
            #. ``verbose in [3, 14]``: Additional messages with job \
               completion status at increasing increasing frequency
            #. ``Verbose >= 15``: prints each job completed as \
               [case].[est].[draw].[fold]

        If ``verbose>=20``, prints to ``sys.stderr``, else ``sys.stdout``.
    """

    def __init__(
            self, scorer, cv=2, shuffle=True, random_state=None,
            error_score=None, metrics=None, array_check=2, verbose=False,
            **kwargs):
        super(Evaluator, self).__init__(**kwargs)

        check_scorer(scorer)
        self.scorer = scorer
        self.scores_ = None

        # TODO: Need to make this accept more than just FoldIndex
        self.cv = cv
        self.indexer = FoldIndex(cv)

        self.shuffle = shuffle
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

    def fit(self, X, y, estimators=None, param_dicts=None,
            n_iter=2, preprocessing=None):
        """Fit

        Fit preprocessing if applicable and evaluate estimators if applicable.
        The method automatically determines whether to only run preprocessing,
        only evaluation (possibly on previously fitted preprocessing), or both.
        Calling ``fit`` will overwrite previously stored data where applicable.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input data to preprocess and create folds from.

        y : array-like, shape=[n_samples, ]
            training labels.

        estimators : list or dict, optional
            set of estimators to use. If no preprocessing is desired or if
            only on preprocessing pipeline should apply to all, pass a list of
            estimators. The list can contain elements of named tuples
            (i.e. ``('my_name', my_est)``).

            If different estimators should be mapped to preprocessing cases,
            a dictionary that maps estimators to each case should
            be passed: ``{'case_a': list_of_est, ...}``.

        param_dicts : dict, optional
            parameter distribution mapping for estimators. Current
            implementation only supports randomized grid search. Passed
            distribution object must have an ``rvs`` method.
            See :mod:`scipy.stats` for details.

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
            class instance with stored estimator evaluation results in
            the ``results`` attribute.
        """
        job = set_job(estimators, preprocessing)
        self._initialize(job, estimators, preprocessing, param_dicts, n_iter)
        self._fit(X, y, job)
        self._get_results()
        return self

    def _initialize(self, job, estimators, preprocessing, param_dicts, n_iter):
        """Set up generators for the job to be performed"""
        if preprocessing and isinstance(preprocessing, list):
            preprocessing = {'pr': preprocessing}

        if 'preprocess' in job:
            self._preprocessing = check_instances(preprocessing)
            self._transformers = make_tansformers(
                sorted(self._preprocessing.items()), self.indexer,
                verbose=max(0, self.verbose - 14))

        if 'evaluate' in job:
            estimators = check_instances(estimators)
            estimators, param_dicts = self._format(estimators, param_dicts)
            self._estimators = estimators

            self.n_iter = n_iter
            self._draw_param_dicts(param_dicts)

            generator = [
                (p_name, l_name, est, i, params)
                for p_name, l_name, est in _flatten(self._estimators)
                for i, params in enumerate(self.params[cat(p_name, l_name)])]

            self._learners = make_learners(
                generator, self.indexer, self.scorer,
                self.error_score, verbose=max(0, self.verbose - 14))

    def _format(self, estimators, param_dicts):
        """Ensure estimator object and param_dict object have right format."""
        preprocessing = self._preprocessing
        if not preprocessing:
            return estimators, param_dicts

        # Set parameter draws for each case
        if isinstance(estimators, list):
            # Cast estimators to all cases
            estimators = {k: [(n, clone(e)) for n, e in estimators]
                          for k in preprocessing}

        # Build params per case
        params = dict()
        for key, pars in param_dicts.items():
            splitted = key.split('.')
            if len(splitted) == 2:
                if splitted[0] not in preprocessing:
                    raise ValueError(
                        "invalid param_dict . Valid keys are "
                        "'case_name.est_name' or 'est_name'. "
                        "Failed on key entry {}.\n"
                        "All keys: {}".format(key, list(preprocessing)))
                params[key] = pars
            else:
                # have an est_name key entry. Need to generate
                for case in preprocessing.keys():
                    key_ = '%s.%s' % (case, key)
                    if key_ in params:
                        # We do not want to overwrite user-specified dists
                        continue
                    params[key_] = pars

        # Quick safety check
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

    def _draw_param_dicts(self, param_dicts):
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
            # 'case_name__est_name'
            if isinstance(self._preprocessing, dict):
                for case in self._preprocessing:
                    for est_name, _ in self._estimators[case]:
                        self._set_params(
                            param_dicts, '%s.%s' % (case, est_name))
            else:
                for est_name, _ in self._estimators:
                    self._set_params(param_dicts, est_name)

    def _get_results(self):
        """For each case-estimator, return best param draw from cv results."""
        data = self.raw_data
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

        self.results = Data(best, decimals=3)

    def _print_prep_start(self):
        """Print preprocessing start and return timer."""
        msg = 'Preprocessing %i preprocessing pipelines over %i CV folds'

        p = len(getattr(self, '_preprocessing', [1]))
        c = self.cv if isinstance(self.cv, int) else self.cv.folds
        return msg % (p, c)

    def _print_eval_start(self):
        """Print initiation message and return timer."""
        preprocessing = getattr(self, '_preprocessing', None)

        if preprocessing is None:

            msg = ('Evaluating %i models for %i parameter draws over %i '
                   'CV folds, totalling %i fits')

            e, c, tot = self._get_count(preprocessing)
            return msg % (e, self.n_iter, c, tot)
        else:

            msg = ('Evaluating %i models for %i parameter draws over %i' +
                   ' preprocessing pipelines and %i CV folds, '
                   'totalling %i fits')

            e, p, c, tot = self._get_count(preprocessing)
            return msg % (e, self.n_iter, p, c, tot)

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

            tot = e * self.n_iter * c
            return int(e), int(p), int(c), int(tot)
