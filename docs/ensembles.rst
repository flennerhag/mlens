.. Documentation of ensembles

.. currentmodule:: mlens.ensemble

Ensemble classes
================

ML-Ensemble implements four types of ensembles:

    * :ref:`super-learner` (stacking)

    * :ref:`subsemble`

    * :ref:`blend-ensemble`

    * :ref:`sequential-ensemble`

Each ensemble class can be built with several layers, and each layer can
output class probabilities if desired. The :class:`SequentialEnsemble` class
is a generic ensemble class that allows the user to mix types between layers,
for instance by setting the first layer to a Subsemble and the second layer
to a Super Learner. Here, we will briefly introduce ensemble specific
parameters and usage. For full documentation, see the :ref:`API` section.

.. _super-learner:

Super Learner
^^^^^^^^^^^^^
The :class:`SuperLearner` (also known as a Stacking Ensemble) is an
supervised ensemble algorithm that uses K-fold estimation to map a
training set :math:`(X, y)` into a prediction set :math:`(Z, y)`,
where the predictions in :math:`Z` are constructed using K-Fold splits
of :math:`X` to ensure :math:`Z` reflects test errors, and that applies
a user-specified meta learner to predict :math:`y` from :math:`Z`.

The main parameter to specify is the ``folds`` parameter that determines the
number of folds to use during cross-validation. The algorithm in
sudo code follows:

    #. Specify a library :math:`L` of base learners

    #. Fit all base learners on :math:`X` and store the fitted estimators.

    #. Split :math:`X` into :math:`K` folds, fit every learner in
       :math:`L` on the training set and predict test set. Repeat until
       all folds have been predicted.

    #. Construct a matrix :math:`Z` by stacking the predictions per fold.

    #. Fit the meta learner on :math:`Z` and store the learner

The ensemble can be used for prediction by mapping a new test set :math:`T`
into a prediction set :math:`Z'` using the learners fitted in (2),
and then mapping :math:`Z'` to :math:`y'` using the fitted meta learner
from (5).

The Super Learner does asymptotically as well as (up to a constant) an
Oracle selector. For the theory behind the Super Learner, see
[#]_ and [#]_ as well as references therein.

Stacking K-fold predictions to cover an entire training set is a time
consuming method and can be prohibitively costly for large datasets.
With large data, other ensembles that fits an ensemble on subsets
can achieve similar performance at a fraction of the training time.
However, when data is noisy or of high variance,
the :class:`SuperLearner` ensure all information is
used during fitting.

References
----------
.. [#] van der Laan, Mark J.; Polley, Eric C.; and Hubbard, Alan E.,
   "Super Learner" (July 2007). U.C. Berkeley Division of Biostatistics
   Working Paper Series. Working Paper 222.
   http://biostats.bepress.com/ucbbiostat/paper222

.. [#] Polley, Eric C. and van der Laan, Mark J.,
   "Super Learner In Prediction" (May 2010). U.C. Berkeley Division of
   Biostatistics Working Paper Series. Working Paper 266.
   http://biostats.bepress.com/ucbbiostat/paper266

Notes
-----
This implementation uses the agnostic meta learner approach, where the
user supplies the meta learner to be used. For the original Super Learner
algorithm (i.e. learn the best linear combination of the base learners),
the user can specify a linear regression as the meta learner.

.. _subsemble:

Subsemble
^^^^^^^^^

:class:`Subsemble` is a supervised ensemble algorithm that uses subsets of
the full data to fit a layer, and within each subset K-fold estimation
to map a training set :math:`(X, y)` into a prediction set :math:`(Z, y)`,
where :math:`Z` is a matrix of prediction from each estimator on each
subset (thus of shape ``[n_samples, (n_partitions * n_estimators)]``).
:math:`Z` is constructed using K-Fold splits of each partition of `X` to
ensure :math:`Z` reflects test errors within each partition. A final
user-specified meta learner is fitted to the final ensemble layer's
prediction, to learn the best combination of subset-specific estimator
predictions.

The main parameters to consider is the number of ``partitions``, which will
increase the number of estimators in the layer by a factor of the number of
base learners specified, and the number of ``folds`` to be used during
cross validation in each partition.

The algorithm in sudo code follows:

    #. For each layer in the ensemble, do:

        #. Specify a library of :math:`L` base learners
        #. Specify a partition strategy and partition :math:`X` into
           :math:`J` subsets.
        #. For each partition do:

            #. Fit all base learners and store them
            #. Create :math:`K` folds
            #. For each fold, do:

               #. Fit all base learners on the training folds
               #. Collect *all* test folds, across partitions, and predict.

        #. Assemble a cross-validated prediction matrix
           :math:`Z \in \mathbb{R}^{(n \times (L \times J))}`  by
           stacking predictions made in the cross-validation step.

    #. Fit the meta learner on :math:`Z` and store the learner.

The ensemble can be used for prediction by mapping a new test set :math:`T`
into a prediction set :math:`Z'` using the learners fitted in
(1.3.1), and then using :math:`Z'` to generate final predictions through
the fitted meta learner from (2).

The Subsemble does asymptotically as well as (up to a constant) the
Oracle selector. For the theory behind the Subsemble, see
[#]_ and references therein.

By partitioning the data into subset and fitting on those, a Subsemble
can reduce training time considerably if estimators does not scale
linearly. Moreover, Subsemble allows estimators to learn different
patterns from each subset, and so can improve the overall performance
by achieving a tighter fit on each subset. Since all observations in the
training set are predicted, no information is lost between layers.

References
----------
.. [#] Sapp, S., van der Laan, M. J., & Canny, J. (2014).
   Subsemble: an  ensemble method for combining subset-specific algorithm
   fits. Journal of Applied Statistics, 41(6), 1247-1259.
   http://doi.org/10.1080/02664763.2013.864263

Notes
-----
This implementation splits X into partitions sequentially, i.e. without
randomizing indices. To achieve randomized partitioning, set ``shuffle``
to ``True``. Supervised partitioning is under development.

.. _blend-ensemble:

Blend Ensemble
^^^^^^^^^^^^^^

The :class:`BlendEnsemble` is a supervised ensemble closely related to
the :class:`SuperLearner`. It differs in that to estimate the prediction
matrix Z used by the meta learner, it uses a subset of the data to predict
its complement, and the meta learner is fitted on those predictions.

The user must specify how much of the data should be used to train the layer,
``test_size``, and how much should be held out for prediction. Prediction for
the held-out set are passed to the next layer or meta estimator, so information
is with each layer.

By only fitting every base learner once on a subset of the full training data,
:class:`BlendEnsemble` is a fast ensemble that can handle very large datasets
simply by only using portion of it at each stage. The cost of this approach is
that information is thrown out at each stage, as one layer will not see the
training data used by the previous layer.

With large data that can be expected to satisfy an i.i.d. assumption, the
:class:`BlendEnsemble` can achieve similar performance to more
sophisticated ensembles at a fraction of the training time. However, with
data data is not uniformly distributed or exhibits high variance the
:class:`BlendEnsemble` can be a poor choice as information is lost at
each stage of fitting.

.. _sequential-ensemble:

Sequential Ensemble
^^^^^^^^^^^^^^^^^^^

The :class:`SequentialEnsemble` allows users to build ensembles with
different classes of layers. Instead of setting parameters upfront during
instantiation, the user specified parameters for each layer when calling
``add``. The user must thus specify what type of layer is being added
(blend, super learner, subsemble), estimators, preprocessing if applicable, and
any layer-specific parameters. The Sequential ensemble is best illustrated
through an example::

    >>> from mlens.ensemble import SequentialEnsemble
    >>> from mlens.metrics.metrics import rmse
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>> from pandas import DataFrame
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = SequentialEnsemble(scorer=rmse)
    >>>
    >>> # Add a subsemble with 10 partitions and 10 folds as first layer
    >>> ensemble.add('subset', [SVR(), Lasso()], n_partitions=10, n_splits=10)
    >>>
    >>> # Add a super learner with 20 folds as second layer
    >>> ensemble.add('stack', [SVR(), Lasso()], n_splits=20)
    >>>
    >>> # Specify a meta estimator
    >>> ensemble.add_meta(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>>
    >>> DataFrame(ensemble.scores_)
                       score_mean  score_std
    layer-1 j0__lasso   11.792905   2.744788
            j0__svr      9.615539   1.185780
            j1__lasso    7.525038   1.235617
            j1__svr      9.164761   0.896510
            j2__lasso    7.239405   1.821464
            j2__svr      9.965071   1.357993
            j3__lasso    9.590788   1.723333
            j3__svr     11.892205   0.880309
            j4__lasso   12.435838   3.475319
            j4__svr      9.368308   0.769086
            j5__lasso   17.357559   2.645452
            j5__svr     11.921103   1.217075
            j6__lasso    8.889963   1.811024
            j6__svr      9.226893   1.030218
            j7__lasso   12.720208   3.521461
            j7__svr     12.751075   1.760458
            j8__lasso   12.178918   1.229540
            j8__svr     12.878269   1.667963
            j9__lasso    7.269251   1.815074
            j9__svr      9.563657   1.214829
    layer-2 lasso        5.660264   2.435897
            svr          8.343091   4.097081

Note how each of the two base learners specified got duplicated to each of the
10 partitions, as denotes by the ``j[num]_`` prefix.
