"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Ensemble transformer class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from ..base import INDEXERS, IdTrain
from ..utils import check_ensemble_build, check_inputs
from ..ensemble.base import BaseEnsemble
from ..externals.sklearn.base import TransformerMixin
from ..externals.sklearn.validation import check_random_state

from ..base import FoldIndex

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

    See Also
    --------
    :class:`SequentialEnsemble`, :class:`Evaluator`

    Parameters
    ----------
    shuffle : bool (default = True)
        whether to shuffle data before generating folds.

    random_state : int (default = None)
        random seed if shuffling inputs.

    scorer : object (default = None)
        scoring function. If a function is provided, base estimators will be
        scored on the training set assembled for fitting the meta estimator.
        Since those predictions are out-of-sample, the scores represent valid
        test scores. The scorer should be a function that accepts an array of
        true values and an array of predictions: ``score = f(y_true, y_pred)``.

    raise_on_exception : bool (default = True)
        whether to issue warnings on soft exceptions or raise error.
        Examples include lack of layers, bad inputs, and failed fit of an
        estimator in a layer. If set to ``False``, warnings are issued instead
        but estimation continues unless exception is fatal. Note that this
        can result in unexpected behavior unless the exception is anticipated.

    sample_dim : int (default = 10)
        dimensionality of training set to sample. During a call to `fit`, a
        random sample of size [sample_dim, sample_dim] will be sampled from the
        training data, along with the dimensions of the training data. If in a
        call to ``transform``, sampling the same indices on the array to
        transform gives the same sample matrix, the transformer will reproduce
        the predictions from the call to ``fit``, as opposed to using the
        base learners fitted on the full training data.

    array_check : int (default = 2)
        level of strictness in checking input arrays.

            - ``array_check = 0`` will not check ``X`` or ``y``

            - ``array_check = 1`` will check ``X`` and ``y`` for
              inconsistencies and warn when format looks suspicious,
              but retain original format.

            - ``array_check = 2`` will impose Scikit-learn array checks,
              which converts ``X`` and ``y`` to numpy arrays and raises
              an error if conversion fails.

    verbose : int or bool (default = False)
        level of verbosity.

            * ``verbose = 0`` silent (same as ``verbose = False``)

            * ``verbose = 1`` messages at start and finish (same as
              ``verbose = True``)

            * ``verbose = 2`` messages for each layer

        If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
        For verbosity in the layers themselves, use ``fit_params``.

    n_jobs : int (default = 1)
        number of CPU cores to use for fitting and prediction.

    Attributes
    ----------
    scores\_ : dict
        if ``scorer`` was passed to instance, ``scores_`` contains dictionary
        with cross-validated scores assembled during ``fit`` call. The fold
        structure used for scoring is determined by ``folds``.

    Examples
    --------
    >>> from mlens.preprocessing import EnsembleTransformer
    >>> from mlens.model_selection import Evaluator
    >>> from mlens.metrics.metrics import rmse
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>> from scipy.stats import uniform
    >>> from pandas import DataFrame
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = EnsembleTransformer()
    >>>
    >>> ensemble.add('stack', [SVR(), Lasso()])
    >>>
    >>> evl = Evaluator(scorer=rmse, random_state=10)
    >>>
    >>> evl.preprocess(X, y, [('scale', ensemble)])
    >>>
    >>> draws = {(None, 'svr'): {'C': uniform(10,  100)},
    ...          (None, 'lasso'): {'alpha': uniform(0.01, 0.1)}}
    >>>
    >>> evl.evaluate(X, y, [SVR(), Lasso()], draws, n_iter=10)
    >>>
    >>> DataFrame(evl.summary)
           fit_time_mean  fit_time_std  test_score_mean  test_score_std  \
    lasso       0.000818      0.000362         7.514181        0.827578
    svr         0.009790      0.000596        10.949149        0.577554
           train_score_mean  train_score_std                      params
    lasso          6.228287         0.949872  {'alpha': 0.0871320643267}
    svr            5.794856         1.348409        {'C': 12.0751949359}
    """

    def __init__(self,
                 shuffle=False,
                 random_state=None,
                 scorer=None,
                 raise_on_exception=True,
                 array_check=2,
                 verbose=False,
                 n_jobs=1,
                 layers=None,
                 backend='multiprocessing',
                 sample_dim=10):

        super(EnsembleTransformer, self).__init__(
                shuffle=shuffle, random_state=random_state,
                scorer=scorer, raise_on_exception=raise_on_exception,
                verbose=verbose, n_jobs=n_jobs, layers=layers,
                backend=backend, array_check=array_check)

        self.sample_dim = sample_dim
        self.id_train = IdTrain(size=sample_dim)

    def add(self, cls, estimators, preprocessing=None, **kwargs):
        """Add layer to ensemble transformer.

        Parameters
        ----------
        cls : str
            layer class. Accepted types are:

                * 'blend' : blend ensemble
                * 'subset' : subsemble
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
            return self._add(estimators, cls, INDEXERS[cls](), preprocessing)

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

        return self._add(estimators=estimators,
                         cls=cls,
                         indexer=indexer,
                         preprocessing=preprocessing,
                         verbose=self.verbose,
                         **kwargs)

    def fit(self, X, y=None):
        """Fit the transformer.

        Same as the fit method on an ensemble, except that a sample of X is
        stored for future comparison.
        """
        X, y = check_inputs(X, y, self.array_check)
        self.id_train.fit(X)
        return super(EnsembleTransformer, self).fit(X, y)

    def predict(self, X):
        """Generate predictions for X. Same as ``transform``."""
        return self.transform(X)

    def transform(self, X, y=None):
        """Transform input :math:`X` into a prediction matrix :math:`Z`.

        If :math:`X`  is the training set, the transformer will
        reproduce the :math:`Z` from the call to ``fit``. If X is another
        data set, :math:`Z` will be produced using base learners fitted on the
        full training data (equivalent to calling ``predict`` on an ensemble.)
        """
        if not self.id_train.is_train(X):
            return super(EnsembleTransformer, self).predict(X)
        else:
            return self._transform(X)

    def _transform(self, X):
        """Reproduce predictions from 'fit' call."""
        if not check_ensemble_build(self):
            # No layers instantiated, but raise_on_exception is False
            return

        X, _ = check_inputs(X, check_level=self.array_check)

        if self.shuffle:
            r = check_random_state(self.random_state)
            idx = r.permutation(X.shape[0])
            X = X[idx]

        y = self.layers.transform(X)

        if y.shape[1] == 1:
            # The meta estimator is treated as a layer and thus a prediction
            # matrix with shape [n_samples, 1] is created. Ravel before return
            y = y.ravel()

        return y
