"""ML-ENSEMBLE

Benchmark of ML-Ensemble against Scikit-learn estimators using Scikit-learn's
friedman1 dataset.

All estimators are instantiated with default settings, and all estimators in
the ensemble are part of the benchmark.

The default ensemble configuration achieves a 25% score improvement as compared
to the best benchmark estimator (GradientBoostingRegressor).

Ensemble training time depends on the number of available cores.

Example output
--------------

Benchmark of ML-ENSEMBLE against Scikit-learn estimators on the friedman1
dataset.

Scoring metric: Root Mean Squared Error.

Available CPUs: 4

Ensemble architecture
Num layers: 1
Meta estimator: GradientBoostingRegressor
layer-1 | Min Max Scaling - Estimators: ['svr'].
layer-1 | Standard Scaling - Estimators: ['elasticnet', 'lasso',
                                          'kneighborsregressor'].
layer-1 | No Preprocessing - Estimators: ['randomforestregressor',
                                          'gradientboostingregressor'].

Benchmark estimators: GBM KNN Kernel Ridge Lasso Random Forest SVR Elastic-Net

Data
Features: 10
Training set sizes: from 2000 to 20000 with step size 2000.

SCORES
  size | Ensemble |      GBM |      KNN | Kern Rid |    Lasso | Random F | ...
  2000 |     0.83 |     0.92 |     2.26 |     2.42 |     3.13 |     1.61 | ...
  4000 |     0.75 |     0.91 |     2.11 |     2.49 |     3.13 |     1.39 | ...
  6000 |     0.66 |     0.83 |     2.02 |     2.43 |     3.21 |     1.29 | ...
  8000 |     0.66 |     0.84 |     1.95 |     2.43 |     3.19 |     1.24 | ...
 10000 |     0.62 |     0.79 |     1.90 |     2.46 |     3.17 |     1.16 | ...
 12000 |     0.68 |     0.86 |     1.84 |     2.46 |     3.16 |     1.10 | ...
 14000 |     0.59 |     0.75 |     1.78 |     2.45 |     3.15 |     1.05 | ...
 16000 |     0.62 |     0.80 |     1.76 |     2.45 |     3.15 |     1.02 | ...
 18000 |     0.59 |     0.79 |     1.73 |     2.43 |     3.12 |     1.01 | ...
 20000 |     0.56 |     0.73 |     1.70 |     2.42 |     4.87 |     0.99 | ...

  size |      SVR |    elNet |
  2000 |     2.32 |     3.18 |
  4000 |     2.31 |     3.16 |
  6000 |     2.18 |     3.25 |
  8000 |     2.09 |     3.24 |
 10000 |     2.03 |     3.21 |
 12000 |     1.97 |     3.21 |
 16000 |     1.92 |     3.20 |
 16000 |     1.87 |     3.19 |
 18000 |     1.83 |     3.17 |
 20000 |     1.81 |     4.75 |

FIT TIMES
  size | Ensemble |      GBM |      KNN | Kern Rid |    Lasso | Random F |
  2000 |     0:01 |     0:00 |     0:00 |     0:00 |     0:00 |     0:00 |
  4000 |     0:02 |     0:00 |     0:00 |     0:00 |     0:00 |     0:00 |
  6000 |     0:04 |     0:00 |     0:00 |     0:01 |     0:00 |     0:00 |
  8000 |     0:06 |     0:00 |     0:00 |     0:04 |     0:00 |     0:00 |
 10000 |     0:08 |     0:01 |     0:00 |     0:08 |     0:00 |     0:00 |
 12000 |     0:09 |     0:01 |     0:00 |     0:12 |     0:00 |     0:00 |
 14000 |     0:12 |     0:01 |     0:00 |     0:20 |     0:00 |     0:00 |
 16000 |     0:16 |     0:02 |     0:00 |     0:34 |     0:00 |     0:00 |
 18000 |     0:20 |     0:02 |     0:00 |     0:47 |     0:00 |     0:00 |
 20000 |     0:25 |     0:02 |     0:00 |     1:20 |     0:00 |     0:00 |

  size |      SVR |    elNet |
  2000 |     0:00 |     0:00 |
  4000 |     0:00 |     0:00 |
  6000 |     0:01 |     0:00 |
  8000 |     0:02 |     0:00 |
 10000 |     0:03 |     0:00 |
 12000 |     0:04 |     0:00 |
 16000 |     0:06 |     0:00 |
 16000 |     0:08 |     0:00 |
 18000 |     0:10 |     0:00 |
 20000 |     0:13 |     0:00 |

"""

import numpy as np
import os

from mlens.ensemble import StackingEnsemble

from mlens.metrics import rmse

from sklearn.datasets import make_friedman1
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.kernel_ridge import KernelRidge

from time import time


def build_ensemble(**kwargs):
    """Generate ensemble."""

    ens = StackingEnsemble(**kwargs)
    prep = {'Standard Scaling': [StandardScaler()],
            'Min Max Scaling': [MinMaxScaler()],
            'No Preprocessing': []}

    est = {'Standard Scaling':
               [ElasticNet(), Lasso(), KNeighborsRegressor()],
           'Min Max Scaling':
               [SVR()],
           'No Preprocessing':
               [RandomForestRegressor(random_state=SEED),
                GradientBoostingRegressor()]}

    ens.add(est, prep)

    ens.add_meta(GradientBoostingRegressor())

    return ens

if __name__ == '__main__':

    print("\nML-ENSEMBLE\n")
    print("Benchmark of ML-ENSEMBLE against Scikit-learn estimators "
          "on the friedman1 dataset.\n")
    print("Scoring metric: Root Mean Squared Error.\n")

    print("Available CPUs: %i\n" % os.cpu_count())

    SEED = 2017
    np.random.seed(SEED)

    step = 4000
    mi = step
    mx = 40000 + step

    ens_multi = build_ensemble(folds=2, shuffle=False, n_jobs=-1)

    ESTIMATORS = {'Ensemble': ens_multi,
                  'Random F': RandomForestRegressor(random_state=SEED,
                                                    n_jobs=-1),
                  '   elNet': make_pipeline(StandardScaler(), ElasticNet()),
                  '   Lasso': make_pipeline(StandardScaler(), Lasso()),
                  'Kern Rid': make_pipeline(MinMaxScaler(), KernelRidge()),
                  '     SVR': make_pipeline(MinMaxScaler(), SVR()),
                  '     GBM': GradientBoostingRegressor(),
                  '     KNN': KNeighborsRegressor(n_jobs=-1)}

    names = {k.strip(' '): k for k in ESTIMATORS}
    times = {e: [] for e in ESTIMATORS}
    scores = {e: [] for e in ESTIMATORS}

    sizes = range(mi, mx, step)

    print('Ensemble architecture')
    print("Num layers: %i" % ens_multi.layers.n_layers)
    print("Meta estimator: %s" % ens_multi.meta_estimator.__class__.__name__)
    for layer in ens_multi.layers.layers:
        for case in ens_multi.layers.layers[layer].preprocessing:
            el = ens_multi.layers.layers['layer-1'].estimators[case]
            print('%s | %s - Estimators: %r.' % (layer, case,
                                                 [e for e, _ in el]))

    print("\nBenchmark estimators", end=": ")
    for name in sorted(names):
        if name == 'Ensemble':
            continue
        print(name, end=" ")
    print('\n')

    print('Data')
    print('Features: %i' % 10)
    print('Training set sizes: from %i to %i with step size %i.\n' % (
          np.floor(mi / 2),
          np.floor((mx - step) / 2),
          np.floor(step/2)))

    print('SCORES')
    print('%6s' % 'size', end=' | ')

    for name in sorted(names):
        print('%s' % names[name], end=' | ')
    print()

    for size in sizes:
        n = int(np.floor(size / 2))

        X, y = make_friedman1(n_samples=size, random_state=SEED)

        print('%6i' % n, end=' | ')
        for name in sorted(names):
            e = clone( ESTIMATORS[names[name]])
            t0 = time()
            e.fit(X[:n], y[:n])
            t1 = time() - t0
            times[names[name]].append(t1)

            s = rmse(e, X[n:], y[n:])
            scores[names[name]].append(s)

            print('%8.2f' % (-s), end=' | ')

        print()

    print('\nFIT TIMES')
    print('%6s' % 'size', end=' | ')

    for name in sorted(names):
        print('%s' % names[name], end=' | ')
    print()

    for i, size in enumerate(sizes):
        n = int(np.floor(size / 2))
        print('%6i' % n, end=' | ')

        for name in sorted(names):

            t = times[names[name]][i]
            m, s = divmod(t, 60)
            print('%5d:%02d' % (m, s), end=' | ')
        print()
