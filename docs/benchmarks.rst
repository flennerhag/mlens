.. _benchmarks:

Performance Benchmarks
======================

The Friedman Regression Problem 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The The Friedman Regression Problem 1, as described in [#]_ and [#]_,
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

.. currentmodule:: mlens.ensemble

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

.. [#] J. Friedman, “Multivariate adaptive regression splines”,
       The Annals of Statistics 19 (1), pages 1-67, 1991.


.. [#] L. Breiman, “Bagging predictors”,
       Machine Learning 24, pages 123-140, 1996.

.. _The Friedman Regression Problem 1: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html#sklearn.datasets.make_friedman1
