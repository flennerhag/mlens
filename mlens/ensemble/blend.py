"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Blend Ensemble class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from .base import BaseEnsemble
from ..index import BlendIndex, FullIndex


class BlendEnsemble(BaseEnsemble):

    r"""Blend Ensemble class.

    The Blend Ensemble is a supervised ensemble closely related to
    the :class:`SuperLearner`. It differs in that to estimate the prediction
    matrix Z used by the meta learner, it uses a subset of the data to predict
    its complement, and the meta learner is fitted on those predictions.

    By only fitting every base learner once on a subset
    of the full training data, :class:`BlendEnsemble` is a fast ensemble
    that can handle very large datasets simply by only using portion of it at
    each stage. The cost of this approach is that information is thrown out
    at each stage, as one layer will not see the training data used by the
    previous layer.

    With large data that can be expected to satisfy an i.i.d. assumption, the
    :class:`BlendEnsemble` can achieve similar performance to more
    sophisticated ensembles at a fraction of the training time. However, with
    data data is not uniformly distributed or exhibits high variance the
    :class:`BlendEnsemble` can be a poor choice as information is lost at
    each stage of fitting.

    See Also
    --------
    :class:`SuperLearner`, :class:`Subsemble`


    .. note :: All parameters can be overriden in the :attr:`add` method unless
        otherwise specified. Notably, the ``backend`` and ``n_jobs`` cannot
        be altered in the :attr:`add` method.

    Parameters
    ----------
    test_size : int, float (default = 0.5)
        the size of the test set for each layer. This parameter can be
        overridden in the :attr:`add` method if different test sizes is desired
        for each layer. If a ``float`` is specified, it is presumed to be the
        fraction of the available data to be used for training, and so
        ``0. < test_size < 1.``.

    shuffle : bool (default = False)
        whether to shuffle data before before processing each layer. This
        parameter can be overridden in the :attr:`add` method if different test
        sizes is desired for each layer.

    random_state : int (default = None)
        random seed for shuffling inputs. Note that the seed here is used to
        generate a unique seed for each layer. Can be overridden in the
        :attr:`add` method.

    scorer : object (default = None)
        scoring function. If a function is provided, base estimators will be
        scored on the prediction made. The scorer should be a function that
        accepts an array of true values and an array of predictions:
        ``score = f(y_true, y_pred)``. Can be overridden in the :attr:`add` method.

    raise_on_exception : bool (default = True)
        whether to issue warnings on soft exceptions or raise error.
        Examples include lack of layers, bad inputs, and failed fit of an
        estimator in a layer. If set to ``False``, warnings are issued instead
        but estimation continues unless exception is fatal. Note that this
        can result in unexpected behavior unless the exception is anticipated.

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

    n_jobs : int (default = -1)
        Degree of parallel processing. Set to -1 for maximum parallelism and
        1 for sequential processing. Cannot be overriden in the :attr:`add` method.

    backend : str or object (default = 'threading')
        backend infrastructure to use during call to
        :class:`mlens.externals.joblib.Parallel`. See Joblib for further
        documentation. To set global backend, set ``mlens.config._BACKEND``.
        Cannot be overriden in the :attr:`add` method.

    model_selection: bool (default=False)
        Whether to use the ensemble in model selection mode. If ``True``,
        this will alter the ``transform`` method. When calling ``transform``
        on new data, the ensemble will call ``predict``, while calling
        ``transform`` with the training data reproduces predictions from the
        ``fit`` call. Hence the ensemble can be used as a pure transformer
        in a preprocessing pipeline passed to the :class:`Evaluator`, as
        training folds are faithfully reproduced as during a ``fit``call and
        test folds are transformed with the ``predict`` method.

    sample_size: int (default=20)
        size of training set sample
        (``[min(sample_size, X.size[0]), min(X.size[1], sample_size)]``)

    Examples
    --------

    Instantiate ensembles with no preprocessing: use list of estimators

    >>> from mlens.ensemble import BlendEnsemble
    >>> from mlens.metrics.metrics import rmse
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = BlendEnsemble()
    >>> ensemble.add([SVR(), ('can name some or all est', Lasso())])
    >>> ensemble.add_meta(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse(y, preds)
    7.3337...


    Instantiate ensembles with different preprocessing pipelines through dicts.

    >>> from mlens.ensemble import BlendEnsemble
    >>> from mlens.metrics.metrics import rmse
    >>> from sklearn.datasets import load_boston
    >>> from sklearn. preprocessing import MinMaxScaler, StandardScaler
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> preprocessing_cases = {'mm': [MinMaxScaler()],
    ...                        'sc': [StandardScaler()]}
    >>>
    >>> estimators_per_case = {'mm': [SVR()],
    ...                        'sc': [('can name some or all ests', Lasso())]}
    >>>
    >>> ensemble = BlendEnsemble()
    >>> ensemble.add(estimators_per_case, preprocessing_cases).add(SVR(),
    ...                                                            meta=True)
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse(y, preds)
    8.249013
    """

    def __init__(
            self, test_size=0.5, shuffle=False, random_state=None, scorer=None,
            raise_on_exception=True, array_check=2, verbose=False, n_jobs=-1,
            backend=None, model_selection=False, sample_size=20, layers=None):
        super(BlendEnsemble, self).__init__(
            shuffle=shuffle, random_state=random_state, scorer=scorer,
            raise_on_exception=raise_on_exception, array_check=array_check,
            verbose=verbose, n_jobs=n_jobs, model_selection=model_selection,
            sample_size=sample_size, layers=layers, backend=backend)

        self.__initialized__ = 0  # Unlock parameter setting
        self.test_size = test_size
        self.__initialized__ = 1  # Protect against param resets

    def add_meta(self, estimator, **kwargs):
        """Meta Learner.

        Compatibility method for adding a meta learner to be used for final
        predictions.

        Parameters
        ----------
        estimator : instance
            estimator instance.

        **kwargs : optional
            optional keyword arguments.
        """
        return self.add(estimators=estimator, meta=True, **kwargs)

    def add(self, estimators, preprocessing=None,
            proba=False, meta=False, propagate_features=None, **kwargs):
        """Add layer to ensemble.

        Parameters
        ----------
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

        proba : bool (default = False)
            Whether to call ``predict_proba`` on base learners.

        propagate_features : list, optional
            List of column indexes to propagate from the input of
            the layer to the output of the layer. Propagated features are
            concatenated and stored in the leftmost columns of the output
            matrix. The ``propagate_features`` list should define a slice of
            the numpy array containing the input data, e.g. ``[0, 1]`` to
            propagate the first two columns of the input matrix to the output
            matrix.

        meta : bool (default = False)
            Whether the layer should be treated as the final meta estimator.

        **kwargs : optional
            optional keyword arguments to instantiate layer with.

        Returns
        -------
        self : instance
            ensemble instance with layer instantiated.
        """
        if meta:
            idx = FullIndex()
        else:
            c = kwargs.pop('test_size', self.test_size)
            idx = BlendIndex(c, raise_on_exception=self.raise_on_exception)

        return super(BlendEnsemble, self).add(
            estimators=estimators, preprocessing=preprocessing, indexer=idx,
            proba=proba, propagate_features=propagate_features, **kwargs)
