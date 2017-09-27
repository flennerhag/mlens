.. _benchmarks:

.. currentmodule:: mlens.ensemble

Performance benchmarks
======================

.. _mnist:

MNIST
^^^^^

MNIST is a standardized image dataset of handwritten digits [#]_, commonly used to benchmark classifiers. Here, we adapt `Scikit-learn's MNIST benchmark`_ to include a supervised :class:`Subsemble`. We use the :class:`sklearn.cluster.MiniBatchKMeans` clustering algorithm to create five partitions that we train using 2-fold cross validation.

Benchmark
---------
We use four base learners, the MLP fitted with Adam, the two random forests and the logistic regression fitted with SAG. Each base learner is predicts with ``predict_proba`` to create a :math:`N \times (10 \times L)` prediction matrix, where :math:`N` is the number of observations and :math:`L` is the number base learners. A :class:`sklearn.ensemble.RandomForestClassifier` is used as meta learner. Here's the code:: 

        clt = MiniBatchKMeans(n_clusters=5, random_state=0)

        ens = Subsemble(partition_estimator=clt,
                        partitions=5,
                        folds=2,
                        verbose=1,
                        n_jobs=-2)

        ens.add(base_learners, proba=True, shuffle=True, random_state=1)
        ens.add_meta(meta, shuffle=True, random_state=2)

The Supervised :class:`Subsemble` outperforms the benchmarks, improving the error rate by about :math:`6\%`. ::

    >>> python mnist.py
    [..]

    Classification performance:
    ===========================
    Classifier               train-time   test-time   error-rate
    ------------------------------------------------------------
    Subsemble                   343.31s       3.17s       0.0210
    MLP_adam                     53.46s       0.11s       0.0224
    Nystroem-SVM                112.97s       0.92s       0.0228
    MultilayerPerceptron         24.33s       0.14s       0.0287
    ExtraTrees                   42.99s       0.57s       0.0294
    RandomForest                 42.70s       0.49s       0.0318
    SampledRBF-SVM              135.81s       0.56s       0.0486
    LinearRegression-SAG         16.67s       0.06s       0.0824
    CART                         20.69s       0.02s       0.1219
    dummy                         0.00s       0.01s       0.8973


The Friedman Regression Problem 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`The Friedman Regression Problem 1`_, as described in [#]_ and [#]_,
is constructed as follows. Set some sample size :math:`m` ,
feature dimensionality :math:`n`, and noise level :math:`e`. Then the input
data :math:`\mathbf{X}` and output data :math:`y(\mathbf{X})` is given by:

.. math::

    \mathbf{X} &= [X_i]_{i \in \{1, 2, ..., n\}} \in
    \mathbb{R}^{m \ \times \ n}, \\
    X &\sim u[0, 1], \\
    \\
    y(\mathbf{X}) &= 10 \sin(\pi X_1 X_2) + 20(X_3 - 0.5)^2 + 10X_4 + 5X_5 +
    \epsilon, \\
    \\
    \epsilon &\sim \mathrm{N}(0, e).

Benchmark
---------

The following benchmark uses 10 features and scores a relatively wide selection
of Scikit-learn estimators against a specified :class:`SuperLearner`. All
estimators are used with default parameter settings. As such, the benchmark
does not reflect the best possible score of each estimator, but shows rather
how stacking even relatively low-performing estimators can yield superior
predictive power. In this case, the Super Learner improves on the best
stand-alone estimator by 25%. ::

    >>> python friedman_scores.py
    Benchmark of ML-ENSEMBLE against Scikit-learn estimators on the friedman1 dataset.

    Scoring metric: Root Mean Squared Error.

    Available CPUs: 4

    Ensemble architecture
    Num layers: 2
    layer-1 | Min Max Scaling - Estimators: ['svr'].
    layer-1 | Standard Scaling - Estimators: ['elasticnet', 'lasso', 'kneighborsregressor'].
    layer-1 | No Preprocessing - Estimators: ['randomforestregressor', 'gradientboostingregressor'].
    layer-2 | (meta) GradientBoostingRegressor

    Benchmark estimators: GBM KNN Kernel Ridge Lasso Random Forest SVR Elastic-Net

    Data
    Features: 10
    Training set sizes: from 2000 to 20000 with step size 2000.

    SCORES
      size | Ensemble |      GBM |      KNN | Kern Rid |    Lasso | Random F |      SVR |    elNet |
      2000 |     0.83 |     0.92 |     2.26 |     2.42 |     3.13 |     1.61 |     2.32 |     3.18 |
      4000 |     0.75 |     0.91 |     2.11 |     2.49 |     3.13 |     1.39 |     2.31 |     3.16 |
      6000 |     0.66 |     0.83 |     2.02 |     2.43 |     3.21 |     1.29 |     2.18 |     3.25 |
      8000 |     0.66 |     0.84 |     1.95 |     2.43 |     3.19 |     1.24 |     2.09 |     3.24 |
     10000 |     0.62 |     0.79 |     1.90 |     2.46 |     3.17 |     1.16 |     2.03 |     3.21 |
     12000 |     0.68 |     0.86 |     1.84 |     2.46 |     3.16 |     1.10 |     1.97 |     3.21 |
     14000 |     0.59 |     0.75 |     1.78 |     2.45 |     3.15 |     1.05 |     1.92 |     3.20 |
     16000 |     0.62 |     0.80 |     1.76 |     2.45 |     3.15 |     1.02 |     1.87 |     3.19 |
     18000 |     0.59 |     0.79 |     1.73 |     2.43 |     3.12 |     1.01 |     1.83 |     3.17 |
     20000 |     0.56 |     0.73 |     1.70 |     2.42 |     4.87 |     0.99 |     1.81 |     4.75 |

    FIT TIMES
      size | Ensemble |      GBM |      KNN | Kern Rid |    Lasso | Random F |      SVR |    elNet |
      2000 |     0:01 |     0:00 |     0:00 |     0:00 |     0:00 |     0:00 |     0:00 |     0:00 |
      4000 |     0:02 |     0:00 |     0:00 |     0:00 |     0:00 |     0:00 |     0:00 |     0:00 |
      6000 |     0:03 |     0:00 |     0:00 |     0:01 |     0:00 |     0:00 |     0:01 |     0:00 |
      8000 |     0:04 |     0:00 |     0:00 |     0:04 |     0:00 |     0:00 |     0:02 |     0:00 |
     10000 |     0:06 |     0:01 |     0:00 |     0:08 |     0:00 |     0:00 |     0:03 |     0:00 |
     12000 |     0:08 |     0:01 |     0:00 |     0:12 |     0:00 |     0:00 |     0:04 |     0:00 |
     14000 |     0:10 |     0:01 |     0:00 |     0:20 |     0:00 |     0:00 |     0:06 |     0:00 |
     16000 |     0:13 |     0:02 |     0:00 |     0:34 |     0:00 |     0:00 |     0:08 |     0:00 |
     18000 |     0:17 |     0:02 |     0:00 |     0:47 |     0:00 |     0:00 |     0:10 |     0:00 |
     20000 |     0:20 |     0:02 |     0:00 |     1:20 |     0:00 |     0:00 |     0:13 |     0:00 |

References
----------

.. [#] Y. LeCun, C. Cortes, C.J.C. Burges "MNIST handwritten digit database",
       http://yann.lecun.com/exdb/mnist/, 2013. 

.. [#] J. Friedman, “Multivariate adaptive regression splines”,
       The Annals of Statistics 19 (1), pages 1-67, 1991.


.. [#] L. Breiman, “Bagging predictors”,
       Machine Learning 24, pages 123-140, 1996.

.. _The Friedman Regression Problem 1: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html#sklearn.datasets.make_friedman1


.. _Scikit-learn's MNIST benchmark: https://github.com/scikit-learn/scikit-learn/blob/master/benchmarks/bench_mnist.py
