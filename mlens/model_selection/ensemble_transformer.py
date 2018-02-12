"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Ensemble transformer class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from ..index import INDEXERS
from ..utils import check_ensemble_build, check_inputs, IdTrain
from ..utils.exceptions import DeprecationWarning
from ..ensemble import BaseEnsemble
from ..externals.sklearn.validation import check_random_state

import warnings


class EnsembleTransformer(BaseEnsemble):

    r"""Ensemble Transformer class.

    The Ensemble class allows users to build layers of an ensemble through a
    transformer API. The transformer is closely related to
    :class:`SequentialEnsemble`, in that any accepted type of layer can be
    added. The transformer differs fundamentally in one significant aspect:
    when fitted, it will store a random sample of the training set together
    with the training dimensions, and if in a call to ``transform``, the
    data to be transformed correspodns to the training set, the transformer
    will recreate the prediction matrix from the ``fit`` call. In contrast,
    a fitted ensemble will only use the base learners fitted on the full
    dataset, and as such predicting the training set will not reproduce the
    predictions from the ``fit`` call.

    The :class:`EnsembleTransformer` is a powerful tool to use as a
    preprocessing pipeline in an :class:`Evaluator` instance, as it would
    faithfully recreate the prediction matrix a potential meta learner would
    face. Hence, a user can 'preprocess' the training data with the
    :class:`EnsembleTransformer` to generate k-fold base learner predictions,
    and then fit different meta learners (or higher-order layers) in a call
    to ``evaluate``.

    Note
    ----
    Will be deprecated in 0.2.2.

    See Also
    --------
    :class:`SequentialEnsemble`, :class:`Evaluator`

    Parameters
    ----------
    shuffle : bool, default = True
        whether to shuffle data before generating folds.

    random_state : int, default = None
        random seed if shuffling inputs.

    raise_on_exception : bool, default = True
        whether to issue warnings on soft exceptions or raise error.
        Examples include lack of layers, bad inputs, and failed fit of an
        estimator in a layer. If set to ``False``, warnings are issued instead
        but estimation continues unless exception is fatal. Note that this
        can result in unexpected behavior unless the exception is anticipated.

    sample_dim : int, default = 20
        dimensionality of training set to sample. During a call to `fit`, a
        random sample of size [sample_dim, sample_dim] will be sampled from the
        training data, along with the dimensions of the training data. If in a
        call to ``transform``, sampling the same indices on the array to
        transform gives the same sample matrix, the transformer will reproduce
        the predictions from the call to ``fit``, as opposed to using the
        base learners fitted on the full training data.

    raise_on_exception : bool, default = True
        whether to issue warnings on soft exceptions or raise error.
        Examples include lack of layers, bad inputs, and failed fit of an
        estimator in a layer. If set to ``False``, warnings are issued instead
        but estimation continues unless exception is fatal. Note that this
        can result in unexpected behavior unless the exception is anticipated.

    array_check : int, default = 2
        level of strictness in checking input arrays. For ``0`` not checks are
        made. For `1``, will check ``X`` and ``y`` for inconsistencies and
        warn when format looks suspicious, but retain original format. With
        ``2``, Scikit-learn's array checks are imposed, which converts
        ``X`` and ``y`` to numpy arrays and raises error if conversion fails.

    verbose : int or bool, default = False
        level of verbosity. ``0`` (or ``False``) is silent, ``1`` (or ``True``)
        outputs messages at start and finish. ``2`` print messages for
        each layer.

    n_jobs : int (default = -1)
        number of CPU cores to use for fitting and prediction.

    backend : str or object (default = 'threading')
        backend infrastructure to use during call to
        :class:`mlens.externals.joblib.Parallel`. See Joblib for further
        documentation. To change global backend, call
        :func:`mlens.config.set_backend()`.

    Examples
    --------
    >>> from mlens.model_selection import EnsembleTransformer
    >>> from mlens.model_selection import Evaluator
    >>> from mlens.metrics import make_scorer, rmse
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>> from scipy.stats import uniform
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = EnsembleTransformer()
    >>>
    >>> ensemble.add('stack', [SVR(), Lasso()])
    >>>
    >>> evl = Evaluator(scorer=make_scorer(rmse), random_state=10)
    >>>
    >>> evl.fit(X, y, preprocessing={'pr': [('scale', ensemble)]})
    >>>
    >>> draws = {('pr', 'svr'):
    ...             {'C': uniform(10,  100)},
    ...          ('pr', 'lasso'):
    ...             {'alpha': uniform(0.01, 0.1)}
    ...         }
    >>>
    >>> evl.fit(X, y, {'pr': [SVR(), Lasso()]}, draws, n_iter=10)
    >>>
    >>> print(evl.results)
                     test_score-m  test_score-s  train_score-m  train_score-s  fit_time-m  fit_time-s  pred_time-m  pred_time-s                           params
    pr  lasso           6.899         0.216          6.100          1.078       0.543       0.126        0.000        0.000      {'alpha': 0.012075194935940151}
    pr  svr            10.164         1.363          5.568          1.575       0.365       0.025        0.003        0.002             {'C': 12.07519493594015}
    """

    def __init__(self,
                 shuffle=False,
                 random_state=None,
                 raise_on_exception=True,
                 array_check=2,
                 verbose=False,
                 n_jobs=-1,
                 layers=None,
                 backend=None,
                 sample_dim=20):
        warnings.warn(
            "EnsembleTransformer is depreciated and will be discontinued in "
            "0.2.2. Use ensemble classes with 'model_selection=True'.",
            DeprecationWarning)

        super(EnsembleTransformer, self).__init__(
                shuffle=shuffle, random_state=random_state,
                raise_on_exception=raise_on_exception,
                verbose=verbose, n_jobs=n_jobs, layers=layers,
                backend=backend, array_check=array_check)

        self.__initialized__ = 0
        self.sample_dim = sample_dim
        self.id_train = IdTrain(size=sample_dim)
        self.__initialized__ = 1

    def add(self, cls, estimators, preprocessing=None, **kwargs):
        """Add layer to ensemble.

        For full set of optional arguments, see the ensemble API for the
        specified type.

        Parameters
        ----------
        cls : str
            layer class. Accepted types are:

                * 'blend' : blend ensemble
                * 'subsemble' : subsemble
                * 'stack' : super learner

        estimators: dict of lists or list or instance
            estimators constituting the layer. If preprocessing is none and the
            layer is meant to be the meta estimator, it is permissible to pass
            a single instantiated estimator. If ``preprocessing`` is
            ``None`` or ``list``, ``estimators`` should be a ``list``.
            The list can either contain estimator instances,
            named tuples of estimator instances, or a combination of both. ::

                option_1 = [estimator_1, estimator_2]
                option_2 = [("est-1", estimator_1), ("est-2", estimator_2)]
                option_3 = [estimator_1, ("est-2", estimator_2)]

            If different preprocessing pipelines are desired, a dictionary
            that maps estimators to preprocessing pipelines must be passed.
            The names of the estimator dictionary must correspond to the
            names of the estimator dictionary. ::

                preprocessing_cases = {"case-1": [trans_1, trans_2],
                                       "case-2": [alt_trans_1, alt_trans_2]}

                estimators = {"case-1": [est_a, est_b],
                              "case-2": [est_c, est_d]}

            The lists for each dictionary entry can be any of ``option_1``,
            ``option_2`` and ``option_3``.

        preprocessing: dict of lists or list, optional (default = None)
            preprocessing pipelines for given layer. If
            the same preprocessing applies to all estimators, ``preprocessing``
            should be a list of transformer instances. The list can contain the
            instances directly, named tuples of transformers,
            or a combination of both. ::

                option_1 = [transformer_1, transformer_2]
                option_2 = [("trans-1", transformer_1),
                            ("trans-2", transformer_2)]
                option_3 = [transformer_1, ("trans-2", transformer_2)]

            If different preprocessing pipelines are desired, a dictionary
            that maps preprocessing pipelines must be passed. The names of the
            preprocessing dictionary must correspond to the names of the
            estimator dictionary. ::

                preprocessing_cases = {"case-1": [trans_1, trans_2],
                                       "case-2": [alt_trans_1, alt_trans_2]}

                estimators = {"case-1": [est_a, est_b],
                              "case-2": [est_c, est_d]}

            The lists for each dictionary entry can be any of ``option_1``,
            ``option_2`` and ``option_3``.

        **kwargs : optional
            optional keyword arguments to instantiate layer with. See
            respective ensemble for further details.

        Returns
        -------
        self : instance
            ensemble instance with layer instantiated.
        """
        if cls not in INDEXERS:
            raise NotImplementedError("Layer class not implemented. Select "
                                      "one of %r." % sorted(INDEXERS))

        # If no kwargs, instantiate with defaults
        if kwargs is None:
            return super(EnsembleTransformer, self).add(
                estimators, INDEXERS[cls](), preprocessing)

        # Else, pop arguments belonging to the indexer
        indexer, kwargs_idx = INDEXERS[cls], dict()

        args = indexer.__init__.__code__.co_varnames
        for arg in args:
            if arg in kwargs:
                kwargs_idx[arg] = kwargs.pop(arg)

        if 'raise_on_exception' in args and \
                'raise_on_exception' not in kwargs_idx:
            kwargs_idx['raise_on_exception'] = self.raise_on_exception
        else:
            kwargs['raise_on_exception'] = kwargs_idx['raise_on_exception']

        indexer = indexer(**kwargs_idx)

        return super(EnsembleTransformer, self).add(
            estimators=estimators, indexer=indexer,
            preprocessing=preprocessing, **kwargs)

    def fit(self, X, y=None, **kwargs):
        """Fit the transformer.

        Same as the fit method on an ensemble, except that a sample of X is
        stored for future training set comparison.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like of shape = [n_samples, ] or None (default = None)
            output vector to trained estimators on.
        """
        X, y = check_inputs(X, y, self.array_check)
        self.id_train.fit(X)
        return super(EnsembleTransformer, self).fit(X, y, **kwargs)

    def transform(self, X, y, **kwargs):
        """Transform input.

        If the input is training set, the transformer will
        reproduce the prediction array from the call to ``fit``. If X is another
        data set, predictions are generated using base learners fitted on the
        full training data (equivalent to calling ``predict`` on an ensemble.)

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like of shape = [n_samples, ] or None (default = None)
            output vector to trained estimators on.
        """
        if self.id_train.is_train(X):
            return self._transform(X, y, **kwargs)
        return self.predict(X, **kwargs), y

    def _transform(self, X, y, **kwargs):
        """Check whether to reproduce predictions from 'fit' call or predict anew."""
        if not check_ensemble_build(self._backend):
            # No layers instantiated, but raise_on_exception is False
            return

        X, y = check_inputs(X, y, check_level=self.array_check)

        if self.shuffle:
            r = check_random_state(self.random_state)
            idx = r.permutation(X.shape[0])
            X = X[idx]
            y = y[idx]

        X = super(EnsembleTransformer, self).transform(X, **kwargs)
        if X.shape[0] != y.shape[0]:
            r = y.shape[0] - X.shape[0]
            y = y[r:]

        return X, y
