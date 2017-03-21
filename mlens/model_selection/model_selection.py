"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Class for parallel tuning a set of estimators that share a common
preprocessing pipeline.
"""

from __future__ import division, print_function

import numpy as np
from pandas import DataFrame, Series

lone_preprocess_cases = None
check_instances = None
preprocess_folds = None
cross_validate = None

from ..utils import print_time, safe_print

from time import time
import sys


class Evaluator(object):

    r"""Model selection across several estimators and preprocessing pipelines.

    The ``Evaluator`` allows users to evaluate several models in one call
    across a set preprocessing pipelines. The class is useful for comparing
    a set of estimators, especially when several preprocessing pipelines is to
    be evaluated. By pre-making all folds and iteratively fitting estimators
    with different parameter settings, array slicing and preprocessing is kept
    to a minimum. This can greatly reduced fit time compared to
    creating pipeline classes for each estimator and pipeline and fitting them
    one at a time in an Scikit-learn ``GridSearch`` class.

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

    preprocessing : dict (default = None)
        dictionary of lists with preprocessing pipelines to fit models on.
        Each pipeline will be used to generate K folds that are stored, hence
        with large data running several cases with many cv folds can require
        considerable memory. ``preprocess`` should be of the form::

                preprocess = {'case-1': [transformer_1, transformer_2],}

    error_score : int, optional
        score to assign when fitting an estimator fails. If ``None``, the
        evaluator will raise an error.

    cv : int or obj (default = 2)
        cross validation folds to use. Either pass a ``KFold`` class
        that obeys the Scikit-learn API.

    shuffle : bool (default = True)
        whether to shuffle input data before creating cv folds.

    random_state : int, optional
        seed for creating folds (if shuffled) and parameter draws

    n_jobs_preprocessing : int (default = -1)
        number of CPU cores to use for preprocessing of folds. ``-1``
        corresponds to all available CPU cores.

    n_jobs_estimators : int (default = -1)
        number of CPU cores to use for preprocessing of folds. ``-1``
        corresponds to all available CPU cores.

    verbose : bool or int (default = False)
        level of printed messages.

    Attributes
    -----------
    summary\_ : DataFrame
        Summary output that shows data for best mean test scores, such as
        test and train scores, std, fit times, and params.

    cv_results\_ : DataFrame
        a table of data from each fit. Includes mean and std of test and train
        scores and fit times, as well as param draw index and parameters.

    best_idx\_ : ndarray
        an array of index keys for best estimator in ``cv_results_``.
    """

    def __init__(self,
                 scorer,
                 preprocessing=None,
                 cv=2,
                 shuffle=True,
                 random_state=None,
                 n_jobs_preprocessing=-1,
                 error_score=None,
                 n_jobs_estimators=-1,
                 verbose=False):

        self.cv = cv
        self.shuffle = shuffle
        self.n_jobs_preprocessing = n_jobs_preprocessing
        self.n_jobs_estimators = n_jobs_estimators
        self.error_score = error_score
        self.random_state = random_state
        self.scorer = scorer
        self.verbose = verbose
        self.preprocessing = check_instances(preprocessing)

    def preprocess(self, X, y):
        """Preprocess folds.

        Method for preprocessing data separately from the evaluation
        method. Helpful if preprocessing is costly relative to
        estimator fitting and several ``evaluate`` calls might be desired.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input data to preprocess and create folds from.

        y : array-like, shape=[n_samples, ]
            training labels.

        Returns
        ----------
        dout : list
            list of lists with folds data. For internal use.
        """
        self.preprocessing_ = clone_preprocess_cases(self.preprocessing)

        if self.verbose > 0:
            printout = sys.stdout if self.verbose >= 50 else sys.stderr
            t0 = self._print_prep_start(self.preprocessing_, printout)

        self.dout = preprocess_folds(self.preprocessing_, X.copy(), y.copy(),
                                     None, self.cv, fit=True, return_idx=False,
                                     shuffle=self.shuffle,
                                     random_state=self.random_state,
                                     n_jobs=self.n_jobs_preprocessing,
                                     verbose=self.verbose)

        if self.verbose > 0:
            print_time(t0, 'Preprocessing done', file=printout)

        return self

    def evaluate(self, estimators, param_dicts, X=None, y=None, n_iter=2,
                 reset_preprocess=False, flush_preprocess=False):
        """Evaluate set of estimators.

        Function for evaluating a set of estimators using cross validation.
        Similar to a randomized grid search, but applies the grid search to all
        specified preprocessing pipelines.

        Parameters
        ----------
        estimators : dict
            set of estimators to use, specified as::

                estimators = {'est-1': estimator_1, 'est-2': estimator_2}

        param_dicts : dict
            param_dicts for estimators. Current implementation only supports
            randomized grid search. Passed distribution object must have a
            ``rvs`` method. See
            :py:class:`sklearn.model_selection.RandomizedSearchCV` for
            details. ``param_dict`` should be specified as::

                param_dicts = {'est-1':
                                   {'param-1': some_distribution,
                                    'param-2': some_distribution},
                               'est-2':
                                   {'param-1': some_distribution,
                                    'param-2': some_distribution},
                               }

        X : array-like of shape = [n_samples, n_features], optional
            input data. If ``preprocess`` was called prior to ``evaluate``
            no data needs to be specified.

        y : array-like, shape=[n_samples, ]
            training labels. If ``preprocess`` was called prior to ``evaluate``
            no data needs to be specified.

        n_iter : int
            number of parameter draws to evaluate.

        reset_preprocess : bool (default = False)
            set to ``True`` to create new preprocessed folds (applicable if
            a ``evaluate`` or ``preprocess`` was call before.

        flush_preprocess : bool (default = False)
            set to ``True`` to drop preprocessed folds after evaluation.
            Useful if memory requirement is large or if the ``Evaluator``
            instance is to be pickled without any arrays.

        Returns
        ---------
        self : instance
            class instance with stored estimator evaluation results.
        """
        self.n_iter = n_iter
        self.estimators_ = check_instances(estimators)
        self.param_dicts_ = param_dicts

        # ===== Preprocess if necessary or requested =====
        if not hasattr(self, 'dout') or reset_preprocess:
            self.preprocess(X, y)

        # ===== Generate n_iter param dictionaries for each estimator =====
        self.param_sets_, self.param_map = self._param_sets()

        # ===== Cross Validate =====
        if self.verbose > 0:
            printout = sys.stdout if self.verbose >= 50 else sys.stderr
            t0 = self._print_eval_start(estimators, self.preprocessing_,
                                        printout)

        out = cross_validate(self.estimators_, self.param_sets_, self.dout,
                             self.scorer, self.error_score,
                             self.n_jobs_estimators, self.verbose)

        # ===== Create summary statistics =====
        self.cv_results_, self.summary_, self.best_idx_ = \
            self._results(out, self.param_map)

        # ===== Job complete =====
        if flush_preprocess:
            del self.dout

        if self.verbose > 0:
            print_time(t0, 'Evaluation done', file=printout)

        return self

    # Auxiliary function for param draws and results mapping
    def _draw_params(self, est_name):
        """Draw a list of param dictionaries for estimator."""
        # Set up empty list of parameter setting
        param_draws = [{} for _ in range(self.n_iter)]

        # Fill list of parameter settings by param
        for param, dist in self.param_dicts_[est_name].items():

            draws = dist.rvs(self.n_iter, random_state=self.random_state)

            for i, draw in enumerate(draws):
                param_draws[i][param] = draw

        return param_draws

    def _param_sets(self):
        """For each estimator, create a mapping of parameter draws."""
        param_sets = {}  # dict with list of param settings for each est
        param_map = {}   # dict with param settings for each est_prep pair

        # Create list of param settings for each estimator
        for est_name, _ in self.estimators_:
            try:
                param_sets[est_name] = self._draw_params(est_name)
            except KeyError:
                # No param draws desired. Set empty dict
                param_sets[est_name] = [{} for _ in range(self.n_iter)]

        # Flatten list to param draw mapping for each preprocessing case
        for est_name, param_draws in param_sets.items():
            for draw, params in enumerate(param_draws):
                for case in self.preprocessing.keys():
                    param_map[(est_name + '-' + case, draw + 1)] = params

        return param_sets, param_map

    @staticmethod
    def _results(out, param_map):
        """Format results into readable pandas DataFrame.

        Parameters
        ----------
        out : list
            list of outputs from ``cross_validate`` call

        param_map : dict
            dictionary of param draws for each estimator.
        """
        # Construct a results DataFrame for each param draw
        out = DataFrame(out, columns=['estimator', 'test_score',
                                      'train_score', 'time',
                                      'param_draw', 'params'])

        # Get mean scores for each param draw
        cv_results = out.groupby(['estimator', 'param_draw']).agg(['mean',
                                                                   'std'])
        cv_results.columns = [tup[0] + '_' + tup[1] for tup in
                              cv_results.columns]

        # Append param settings
        param_map = Series(param_map)
        param_map.index.names = ['estimator', 'param_draw']
        cv_results['params'] = param_map.loc[cv_results.index]

        # Create summary table of best scores
        ts_id = 'test_score_mean'
        best_score = cv_results.loc[:, ts_id].groupby(level=0).apply(np.argmax)
        best_idx = best_score.values
        summary = cv_results.loc[best_idx].reset_index(1, drop=True)
        summary.sort_values(by=ts_id, ascending=False, inplace=True)

        return cv_results, summary, best_idx

    def _print_prep_start(self, preprocessing, printout):
        """Print preprocessing start and return timer."""
        t0 = time()
        msg = 'Preprocessing %i preprocessing pipelines over %i CV folds'

        try:
            p = max(len(preprocessing), 1)
        except Exception:
            p = 0

        c = self.cv if isinstance(self.cv, int) else self.cv.n_splits

        safe_print(msg % (p, c), file=printout)
        return t0

    def _print_eval_start(self, estimators, preprocessing, printout):
        """Print initiation message and return timer."""
        t0 = time()
        msg = ('Evaluating %i models for %i parameter draws over %i' +
               ' preprocessing pipelines and %i CV folds, totalling %i fits')
        e = len(estimators)
        try:
            p = max(len(preprocessing), 1)
        except Exception:
            p = 0

        c = self.cv if isinstance(self.cv, int) else self.cv.n_splits

        tot = e * max(1, p) * self.n_iter * c
        safe_print(msg % (e, self.n_iter, p, c, tot), file=printout)
        return t0
