"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Sequential Ensemble class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from .base import BaseEnsemble
from ..index import INDEXERS
from ..utils import kwarg_parser


class SequentialEnsemble(BaseEnsemble):

    r"""Sequential Ensemble class.

    The Sequential Ensemble class allows users to build ensembles with
    different classes of layers. The type of layer and its parameters are
    specified when added to the ensemble. See respective ensemble class for
    details on parameters.

    See Also
    --------
    :class:`BlendEnsemble`, :class:`Subsemble`, :class:`SuperLearner`

    Parameters
    ----------
    shuffle : bool (default = False)
        whether to shuffle data before before processing each layer.
        For greater control, specify ``shuffle`` when adding the layer.

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
        number of CPU cores to use for fitting and prediction.

    backend : str or object (default = 'threading')
        backend infrastructure to use during call to
        :class:`mlens.externals.joblib.Parallel`. See Joblib for further
        documentation. To change global backend, set
        ``mlens.config._BACKEND``

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
    >>> from mlens.ensemble import SequentialEnsemble
    >>> from mlens.metrics.metrics import rmse
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = SequentialEnsemble()
    >>>
    >>> # Add a subsemble with 5 partitions as first layer
    >>> ensemble.add('subsemble', [SVR(), Lasso()], partitions=10, folds=10)
    >>>
    >>> # Add a super learner as second layer
    >>> ensemble.add('stack', [SVR(), Lasso()], folds=20)
    >>>
    >>> # Specify a meta estimator
    >>> ensemble.add_meta(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse(y, preds)
    6.5628...
    """

    def __init__(
            self, shuffle=False, random_state=None, scorer=None,
            raise_on_exception=True, array_check=2, verbose=False, n_jobs=-1,
            backend=None, model_selection=False, sample_size=20, layers=None):
        super(SequentialEnsemble, self).__init__(
            shuffle=shuffle, random_state=random_state, scorer=scorer,
            raise_on_exception=raise_on_exception, verbose=verbose,
            n_jobs=n_jobs, layers=layers, array_check=array_check,
            model_selection=model_selection, sample_size=sample_size,
            backend=backend)

    def add_meta(self, estimator, **kwargs):
        """Meta Learner.

        Meta learner to be used for final predictions.

        Parameters
        ----------
        estimator : instance
            estimator instance.

        **kwargs : optional
            optional keyword arguments.
        """
        return self.add(cls='full', estimators=estimator, meta=True, **kwargs)

    def add(self, cls, estimators, preprocessing=None, meta=False, **kwargs):
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

        if cls == 'subsemble' and 'partition_estimator' in kwargs:
            cls = 'clusteredsubsemble'

        # instantiate the indexer
        indexer = INDEXERS[cls]
        kwargs_idx, kwargs = kwarg_parser(indexer.__init__, kwargs)
        indexer = indexer(**kwargs_idx)

        return super(SequentialEnsemble, self).add(
            estimators=estimators, indexer=indexer,
            preprocessing=preprocessing, **kwargs)
