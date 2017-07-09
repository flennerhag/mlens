.. _ensemble-tutorial:

Tutorials
=========

The following tutorials highlight advanced functionality and a more detailed
view of how ensembles are built and trained.

===============================  ==============================================
Tutorial                         Content
===============================  ==============================================
:ref:`propa-tutorial`            Propagate feature input features through layers
\                                to allow several layers to see the same input.
:ref:`proba-tutorial`            Build layers that output class probabilities from each base
\                                learner so that the next layer or meta estimator learns
\                                from probability distributions.
:ref:`sequential-tutorial`       How to build ensembles with different layer classes
:ref:`memory-tutorial`           Avoid loading data into the parent process by specifying a
\                                file path to a memmaped array or a csv file.
:ref:`model-selection-tutorial`  Build transformers that replicate layers in ensembles for
\                                model selection of higher-order layers and / or meta learners.
===============================  ==============================================

We use the same preliminary settings as in the
:ref:`getting started <getting-started>` section.


.. _proba-tutorial:

Propagating input features
--------------------------

When stacking several layers of base learners, the variance of the input
will typically get smaller as learners get better and better at predicting
the output and the remaining errors become increasingly difficult to correct
for. This multicolinearity can significantly limit the ability of the
ensemble to improve upon the best score of the subsequent layer as there is too
little variation in predictions for the ensemble to learn useful combinations.
One way to increase this variation is to propagate features from the original
input and / or earlier layers. To achieve this in ML-Ensemble, we use the
``propagate_features`` attribute. To see how this works, let's compare the
a three-layer ensemble with and without feature propagation. ::

    from mlens.ensemble import SuperLearner
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    def build_ensemble(incl_meta, propagate_features=None):
        """Return an ensemble."""
        if propagate_features:
            n = len(propagate_features)
            propagate_features_1 = propagate_features
            propagate_features_2 = [i for i in range(n)]
        else:
            propagate_features_1 = propagate_features_2 = None

        estimators = [RandomForestClassifier(random_state=seed), SVC()]

        ensemble = SuperLearner()
        ensemble.add(estimators, propagate_features=propagate_features_1)
        ensemble.add(estimators, propagate_features=propagate_features_2)

        if incl_meta:
            ensemble.add_meta(LogisticRegression())
        return ensemble

Without feature propagation, the meta learner will learn from the predictions
of the penultimate layers::

    >>> base = build_ensemble(False)
    >>> base.fit(X, y)
    >>> base.predict(X)
    array([[ 2.,  2.],
           [ 2.,  2.],
           [ 2.,  2.],
           [ 1.,  1.],
           [ 1.,  1.]])

When we propagate features, some (or all) of the input seen by one layer is
passed along to the next layer. For instance, we can propagate some or all of
the input array through our two intermediate layers to the meta learner input
of the meta learner::

    >>> base = build_ensemble(False, [1, 3])
    >>> base.fit(X, y)
    >>> base.predict(X)
    array([[ 3.20000005,  2.29999995,  2.        ,  2.        ],
           [ 3.20000005,  2.29999995,  2.        ,  2.        ],
           [ 3.        ,  2.0999999 ,  2.        ,  2.        ],
           [ 3.20000005,  1.5       ,  1.        ,  1.        ],
           [ 2.79999995,  1.39999998,  1.        ,  1.        ]])

This meta learner will not see the predictions made by the penultimate layer,
as well as the second and fourth feature of the input array. By propagating
features, the issue of multicolinearity in deep ensembles can be mitigated.
In particular, it can give the meta learner greater opportunity to identify
neighborhoods in the original feature space where base learners struggle. We
can get an idea of how feature propagation works with our toy example. First,
we need a simple ensemble evaluation routine. ::

    def evaluate_ensemble(propagate_features):
        """Wrapper for ensemble evaluation."""
        ens = build_ensemble(True, propagate_features)
        ens.fit(X[:75], y[:75])
        pred = ens.predict(X[75:])
        return f1_score(pred, y[75:], average='micro')

In our case, propagating the original features through two layers of the same
library of base learners gives a dramatic increase in performance on the test
set:

    >>> score_no_prep = evaluate_ensemble(None)
    >>> score_prep = evaluate_ensemble([0, 1, 2, 3])
    >>> print("Test set score no feature propagation  : %.3f" % score_no_prep)
    >>> print("Test set score with feature propagation: %.3f" % score_prep)

By combining feature propagation with the ``mlens.preprocessing.Subset``
transformer, one can propagate the feature through several layers without
any of the base estimators in those layers seeing the propagated features. This
can be desirable if you want to propagate the input features to the meta
learner, but don't want the intermediate base learners (in layers 2, 3, ...) to
always have the original data as input. In this case, one specified propagation
as we did above, but add a preprocessing pipeline to intermediate layers::

        from mlens.preprocessing import Subset

        estimators = [RandomForestClassifier(random_state=seed), SVC()]
        ensemble = SuperLearner()

        # Initial layer, propagate as before
        ensemble.add(estimators, propagate_features=[0, 1])

        # Intermediate layer, keep propagating, but add a preprocessing
        # pipeline that selects a subset of the input
        ensemble.add(estimators,
                     preprocessing=[Subset([2, 3])],
                     propagate_features=[0, 1])

In the above example, the two first features of the original input data
will be propagated through both layers, but the second layer will not be
trained on it. Instead, it will only see the predictions made by the base
learners in the first layer. ::

    >>> ensemble.fit(X, y)
    >>> n = ensemble.layer_2.estimators_[0][1][1].feature_importances_.shape[0]
    >>> m = ensemble.predict(X).shape[1]
    >>> print("Num features seen by estimators in intermediate layer: %i" % n)
    >>> print("Num features in the output array of the intermediate layer: %i" % m)
    Num features seen by estimators in intermediate layer: 2
    Num features in the output array of the intermediate layer: 4

.. _proba-tutorial:

Probabilistic ensemble learning
-------------------------------

When the target to predict is a class label, it can often be beneficial to
let higher-order layers or the meta learner learn from *class probabilities*,
as opposed to the predicted class. Scikit-learn classifiers can return a
matrix that, for each observation in the test set, gives the probability that
the observation belongs to the a given class. While we are ultimately
interested in class membership, this information is much richer that just
feeding the predicted class to the meta learner. In essence, using class
probabilities allow the meta learner to weigh in not just the predicted
class label (the highest probability), but also with what confidence each
estimator makes the prediction, and how estimators consider the alternative.

First, let us set a benchmark ensemble performance when learning is by
predicted class membership. ::

    from mlens.ensemble import BlendEnsemble
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    def build_ensemble(proba, **kwargs):
        """Return an ensemble."""
        estimators = [RandomForestClassifier(random_state=seed),
                      SVC(probability=proba)]

        ensemble = BlendEnsemble(**kwargs)
        ensemble.add(estimators, proba=proba)   # Specify 'proba' here
        ensemble.add_meta(LogisticRegression())

        return ensemble

As in the :ref:`ensemble guide <ensemble-guide>`, we fit on the first half,
and test on the remainder. ::

    >>> ensemble = build_ensemble(proba=False)
    >>> ensemble.fit(X[:75], y[:75])
    >>> preds = ensemble.predict(X[75:])
    >>> f1_score(preds, y[75:], average='micro')
    0.69333333333333336

Not particularly impressive. Recall that the blend ensemble consumes
observation between layers; in this case, each layer sees only half of the
samples.

To enable probibalistic learning, we set ``proba=True`` in the ``add``
method. Note that when a layer is declared as a meta learner (either through
the ``add_meta`` method or by setting ``meta=True`` in the ``add`` method),
the layer will always predict classes. ::

    >>> ensemble = build_ensemble(proba=True)
    >>> ensemble.fit(X[:75], y[:75])
    >>> preds = ensemble.predict(X[75:])
    >>> print('Prediction shape: {}'.format(preds.shape))
    >>> f1_score(preds, y[75:], average='micro')
    Prediction shape: (75,)
    0.97333333333333338

In this case, using probabilities had a drastic effect on predictive
performance, increasing some 40%. As a final remark, even though the base
learners predict probabilities, the meta layer returns predictions. If you
want an ensemble to return the matrix of predicted probabilities, avoid
specifying a meta layer.

.. _sequential-tutorial:

General multi-layer ensemble learning
-------------------------------------

.. currentmodule:: mlens.ensemble

The modular ``add`` API of ML-Ensembles allow users to build arbitrarily
deep ensembles. If you would like to alternate between the *type* of each layer
the :class:`SequentialEnsemble` class can be used to specify what type of
layer (i.e. stacked, blended, subsamle-style) to add. This can be particularly
powerful if facing a large dataset, as the first layer can use a fast appraoch
such as blending, while subsequent layers fitted on the remaining data can
use more computationally intensive approaches. The type of layer, along with
any parameter settings pertaining to that layer, are specified in the
``add`` method. ::

    from mlens.ensemble import SequentialEnsemble

    ensemble = SequentialEnsemble()

    # The initial layer is a the same as a BlendEnsemble with one layer
    ensemble.add('blend', [SVC(), RandomForestClassifier(random_state=seed)])

    # The second layer is a the same as a SuperLearner with one layer
    ensemble.add('stack', [SVC(), RandomForestClassifier(random_state=seed)])

    # The meta estimator is added as in any other ensemble
    ensemble.add_meta(SVC())

Note that currently, the sequential ensemble uses the backend terminology and
may not overlap with what the ensemble classes uses. This will be fixed in a
coming release. Until then, the following conversion may be helpful.

===================  ============================
front-end parameter  SequentialEnsemble parameter
===================  ============================
'SuperLearner'       'stack'
'BlendEnsemble'      'blend'
'Subsemble'          'subset'
'folds'              'n_splits'
'partitions'         'n_partitions'
===================  ============================

This ensemble can now be used for fitting and prediction with the conventional
syntax. ::

    >>> preds = ensemble.fit(X[:75], y[:75]).predict(X[75:])
    >>> f1_score(preds, y[75:], average='micro')
    0.97333333333333338

In this case, the multi-layer :class:`SequentialEnsemble` with an initial
blended layer and second stacked layer achieves similar performance as the
:class:`BlendEnsemble` with probabilistic learning. Note that we could have
made any of the layers probabilistic by setting ``Proba=True``.


.. _memory-tutorial:

Passing file paths as data input
--------------------------------

With large datasets, it can be expensive to load the full data into memory as
a numpy array. Since ML-Ensemle uses a memmaped cache, the need to keep the
full array in memory can be entirely circumvented by passing a file path as
entry to ``X`` and ``y``. There are two important things to note when doing
this.

First, ML-Ensemble delpoys Scikit-learn's array checks, and passing a
string will cause an error. To avoid this, the ensemble must be initialized
with ``array_check=0``, in which case there will be no checks on the array.
The user should make certain that the the data is approprate for esitmation,
by converting missing values and infinites to numerical representation,
ensuring that all features are numerical, and remove any headers,
index columns and footers.

Second, ML-Ensemble expects the file to be either a ``csv``,
an ``npy`` or ``mmap`` file and will treat these differently.

    - If a path to a ``csv`` file is passed, the ensemble will first **load**
      the file into memory, then dump it into the cache, before discarding the
      file from memory by replacing it with a pointer to the memmaped file.
      The loading module used for the ``csv``
      file is the :func:`numpy.loadtxt` function.

    - If a path to a ``npy`` file is passed, a memmaped pointer to it will be
      loaded.

    - If a path to a ``mmap`` file is passed, it will be used as the memmaped
      input array for estimation.

    ::

        import os
        import gc
        import tempfile

        # We create a temporary folder in the current working directory
        temp = tempfile.TemporaryDirectory(dir=os.getcwd())

        # Dump the X and y array in the temporary directory, here as csv files
        fx = os.path.join(temp.name, 'X.csv')
        fy = os.path.join(temp.name, 'y.csv')

        np.savetxt(fx, X)
        np.savetxt(fy, y)


We can now fit any ensemble simply by passing the file pointers ``fx`` and
``fy``. Remember to set ``array_check=0``. ::

    >>> ensemble = build_ensemble(False, array_check=0)
    >>> ensemble.fit(fx, fy)
    >>> preds = ensemble.predict(fx)
    >>> preds[:10]
    array([ 2.,  2.,  2.,  1.,  1.,  2.,  2.,  2.,  2.,  2.])

If you are following the examples on your machine,
don't forget to remove the temporary directory.

::

    try:
        temp.cleanup()
        del temp
    except OSError:
        # This can fail on Windows
        pass

.. _model-selection-tutorial:

Ensemble model selection
------------------------

Ensembles benefit from a diversity of base learners, but often it is not clear
how to parametrize the base learners. In fact, combining base learners with
lower predictive power can often yield a superior ensemble. This hinges on the
errors made by the base learners being relatively uncorrelated, thus allowing
a meta estimator to learn how to overcome each model's weakness. But with
highly correlated errors, there is little for the ensemble to learn from.

To fully exploit the learning capacity in an ensemble, it is beneficial to
conduct careful hyper parameter tuning, treating the base learner's parameters
as the parameters of the ensemble. By far the most critical part of the
ensemble is the meta learner, but selecting an appropriate meta learner can be
an ardous task if the entire ensemble has to be evaluated each time.

.. py:currentmodule:: mlens.preprocessing

The :class:`EnsembleTransformer` can be leveraged to treat the initial
layers of the ensemble as preprocessing. Thus, a copy of the transformer is
fitted once on each fold, and any model selection will use these pre-fits to
convert raw input to prediction matrices that corresponds to the output of the
specified ensemble.

.. py:currentmodule:: mlens.ensemble

The transformer follows the same API as the :class:`SequentialEnsemble`, but
does not implement a meta estimator and has a transform method that recovers
the prediction matrix from the ``fit`` call. In the following example,
we run model selection on the meta learner of a blend ensemble, and try
two configurations of the blend ensemble: learning from class predictions or
from probability distributions over classes. ::

    from mlens.preprocessing import EnsembleTransformer
    from mlens.model_selection import Evaluator
    from scipy.stats import uniform, randint
    from pandas import DataFrame

    # Set up two competing ensemble bases as preprocessing transformers:
    # one blend ensemble base with proba and one without
    base_learners = [RandomForestClassifier(random_state=seed),
                     SVC(probability=True)]

    proba_transformer = EnsembleTransformer().add('blend', base_learners, proba=True)
    class_transformer = EnsembleTransformer().add('blend', base_learners, proba=False)

    # Set up a preprocessing mapping
    # Each pipeline in this map is fitted once on each fold before
    # evaluating candidate meta learners.
    preprocessing = {'proba': [('layer-1', proba_transformer)],
                     'class': [('layer-1', class_transformer)]}

    # Set up candidate meta learners
    # We can specify a dictionary if we wish to try different candidates on
    # different cases, or a list if all estimators should be run on all
    # preprocessing pipelines (as in this example)
    meta_learners = [SVC(), ('rf', RandomForestClassifier(random_state=2017))]

    # Set parameter mapping
    # Here, we differentiate distributions between cases for the random forest
    params = {'svc': {'C': uniform(0, 10)},
              ('class', 'rf'): {'max_depth': randint(2, 10)},
              ('proba', 'rf'): {'max_depth': randint(2, 10),
                                'max_features': uniform(0.5, 0.5)}
              }

    evaluator = Evaluator(scorer=f1, random_state=2017, cv=20)

    evaluator.fit(X, y, meta_learners, params, preprocessing=preprocessing, n_iter=2)

We can now compare the performance of the best fit for each candidate
meta learner. ::

    >>> DataFrame(evaluator.summary)
               test_score_mean  test_score_std  train_score_mean  train_score_std  fit_time_mean  fit_time_std                                             params
    class rf          0.955357        0.060950          0.972535         0.008303       0.024585      0.014300                                   {'max_depth': 5}
          svc         0.961607        0.070818          0.972535         0.008303       0.000800      0.000233                               {'C': 7.67070164682}
    proba rf          0.980357        0.046873          0.992254         0.007007       0.022789      0.003296   {'max_depth': 3, 'max_features': 0.883535082341}
          svc         0.974107        0.051901          0.969718         0.008060       0.000994      0.000367                              {'C': 0.209602254061}


In this toy example, our model selection suggests the Random Forest is the
best meta learner when the ensemble uses probabilistic learning.
