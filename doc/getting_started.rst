.. Some stuff on getting started

.. _getting-started:

User Guides
===========

ML-Ensemble follows closely the Scikit-learn_ API, and any ML-Ensemble
estimator behaves almost identically to a Scikit-learn estimator.
To quickly get a feel for a an ML-Ensemble estimator behaves, see
the :ref:`ensemble-guide` on how to instantiate, fit and predict with an
ensemble. The :ref:`model-selection-guide` shows how to use the model selection
library, while the :ref:`visualization-guide` gives an introduction to the
plotting functionality. For a full tutorial on how to use ML-Ensemble as a
pipeline for ensemble learning, see the :ref:`ensemble-tutorial`.

Preliminaries
-------------

All guides use the same data and have the same dependencies. To avoid writing
the same imports several times, we put all common imports here. ::

    import numpy as np

    # Set seed for reproducibility
    seed = 2017
    np.random.seed(seed)

    # We will use the f1 scorer
    from mlens.metrics import make_scorer
    from sklearn.metrics import f1_score

    def f1(y_true, y_pred):
    """Wrapper around f1_scorer with average='micro'."""
        return f1_score(y_true, y_pred, average='micro')

    # We use the iris data set
    from sklearn.datasets import load_iris

    data = load_iris()
    idx = np.random.permutation(150)
    X = data.data[idx]
    y = data.target[idx]

    # We induce non-uniform noise to make the problem more challenging
    for i in range(X.shape[1]):
        X[:, i] += np.random.chisquare(1, X.shape[0]) * i

.. _ensemble-guide:

Ensemble Guide
--------------

The only fundamental difference between a ML-Ensemble estimator and a
Scikit-learn estimator is that an ML-Ensemble estimator usually requires the
user to specify one or several layers of estimators, and if applicable a final
meta estimator. Here, we will build a one-layer stacking ensemble that combines
the predictions of a `Random Forest`_ and a `Support Vector Machine`_ through a
`Logistic regression`_. ::

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    from mlens.ensemble import StackingEnsemble

    # Instantiate the ensemble with the f1 scorer
    # Passing a scorer will create cv scores for every estimator in all layers
    ensemble = StackingEnsemble(scorer=f1, random_state=seed)

    # Build the first layer
    ensemble.add([RandomForestClassifier(random_state=seed), SVC()])

    # Attach the final meta estimator
    ensemble.add_meta(LogisticRegression())

    # Fit ensemble on half the data
    ensemble.fit(X[:75], y[:75])

    # Predict the other half
    preds = ensemble.predict(X[75:])


We can now check how well each estimator in the layers of the ensemble::

    >>> ensemble.scores_
    {'layer-1--svc': 0.47999999999999998,
     'layer-1--randomforestclassifier': 0.64000000000000001}

To round off, let's see how the ensemble as a whole fared. ::

    >>> f1(preds, y[75:])
    0.66666666666666663

.. _model-selection-guide:

Model Selection Guide
---------------------

The model selection suite is constantly expanding, so make sure to check in
regularly! The work horse is the ``Evaluator`` class, that allows a user to
evaluate several models in one go, thus avoiding fitting the same preprocessing
pipelines time and again. Moreover, if the data is large, avoiding repeated
slicing for creating folds rapidly amounts to significant time saved.

Let's evaluate how a `Naive Bayes`_ model and a `K-Nearest-Neighbor`_ model
performs under three different preprocessing scenarios: non preprocessing at
all, standard scaling, and subset selection. In this latter scenario, we will
simply stipulate the models use the first two columns of ``X``. ::

    from mlens.model_selection import Evaluator
    from mlens.preprocessing import StandardScaler, Subset

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier

    from scipy.stats import randint

    # Map preprocessing cases through a dictionary
    preprocess_cases = {'none': [],
                        'sc': [StandardScaler()],
                        'sub': [Subset([0, 1])]}

    # Instantiate the evaluator
    evaluator = Evaluator(scorer, preprocess_cases,
                          cv=10, random_state=seed, verbose=1)

Once the preprocessing is set up and the evaluator is instantiated, we can
pre-make the cv folds for each preprocessing case if we wish to separate
out the preprocessing and the actual evaluation. This can make sense if the
preprocessing is time-consuming, for instance if the preprocessing
constitutes the base of an ensemble
(XXX: need to set up the EnsembleTransformers). We can achieve this by
calling the ``preprocess`` method::

    >>> evaluator.preprocess(X, y)
    Preprocessing 3 preprocessing pipelines over 10 CV folds
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:    0.0s finished
    Preprocessing done | 00:00:00

To launch an evaluation, we need a mapping of parameter distributions and
a list of estimators. It is important that the name entries in the
parameter distribution mapping corresponds to the names of the estimators. If
estimators are left unnamed, i.e. as a list of estimators
``[est_1, est_2]``, these will be given the name of their class in lower
letters. So the ``Lasso`` estimator will be named ``lasso``. Alternatively, you
can pass a named tuple ``(name, est)`` instead of only the estimator instance,
if you wish to directly control the name of the estimator. ::

    # The Gaussian model has no interesting parameters to tune, se we leave it
    # out. We will rename the KNeighborsClassifier to 'knn' for simplicity.
    params = {'knn':
                {'n_neighbors': randint(2, 20)}}

    # We must rename the K-Nearest-Neighbor estimator
    # to 'knn' to match the entry in the 'params' dict.
    estimators = [('gnb', GaussianNB()), ('knn', KNeighborsClassifier())]

To evaluate, call the ``evaluate`` method. If preprocessing folds have
already been generated, there is no need passing ``X`` and ``y`` again.
Make sure to specify how many parameter draws you with to evaluate
(the ``n_iter`` parameter). ::

    >>> evaluator.evaluate(estimators, params, n_iter=10)
    Evaluating 2 models for 10 parameter draws over 3 preprocessing pipelines and 10 CV folds, totalling 600 fits
    [Parallel(n_jobs=-1)]: Done 600 out of 600 | elapsed:    1.0s finished
    Evaluation done | 00:00:01

The results for all parameter draws are stored in ``cv_results_``. The
``summary_`` attribute contains the best parameter setting for each estimator
in each preprocessing case. Calling ``evaluator.summary_`` gives the following
table:

=========  ===============  ==============  ================  ===============  =========  ========  ==================
estimator  test_score_mean  test_score_std  train_score_mean  train_score_std  time_mean  time_std  params
=========  ===============  ==============  ================  ===============  =========  ========  ==================
knn-sub    0.720000         0.159629        0.782963           0.024949        0.000555   0.000326  {'n_neighbors': 9}
knn-sc     0.720000         0.132591        0.783704           0.020893        0.000626   0.000405  {'n_neighbors': 8}
gnb-sub    0.713333         0.169385        0.702963           0.014999        0.001243   0.000723                  {}
gnb-none   0.706667         0.095323        0.748148           0.009877        0.001060   0.000433                  {}
gnb-sc     0.706667         0.095323        0.748148           0.009877        0.001778   0.001728                  {}
knn-none   0.693333         0.114180        0.804444           0.014055        0.000803   0.000538  {'n_neighbors': 5}
=========  ===============  ==============  ================  ===============  =========  ========  ==================

So we can quickly surmise that the K-Nearest-Neighbor estimator does generally
better than the Naive Bayes estimator. For the KNN, wisely choosing a subset
(here, those with least induced noise) and standardizing the data were equally
efficient preprocessing pipelines.

.. _visualization-guide:

Visualization Guide
-------------------



.. _Scikit-learn:  http://scikit-learn.org/stable/
.. _Random Forest: https://en.wikipedia.org/wiki/Random_forest
.. _Support Vector Machine: https://en.wikipedia.org/wiki/Support_vector_machine
.. _Logistic regression: https://en.wikipedia.org/wiki/Logistic_regression
.. _Naive Bayes: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
.. _K-Nearest-Neighbor: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
