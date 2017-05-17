"""ML-ENSEMBLE

Comparison of multithreading performance as dataset scale increases.

Example Output
--------------

ML-ENSEMBLE

Threading performance test for data set dimensioned up to (1000000, 10)
Available CPUs: 4

Ensemble architecture
Num layers: 2
Fit per base layer estimator: 4 + 1
layer-1 | Estimators: 4x Lasso
layer-2 | Meta Estimator: Lasso

FIT TIMES
samples | cores: 1 | cores: 2 | cores: 4 |


"""

import numpy as np

import os
from mlens.base import FoldIndex
from mlens.ensemble.base import LayerContainer
from mlens.ensemble import SuperLearner, Subsemble, BlendEnsemble
from mlens.utils import print_time

from sklearn.datasets import make_friedman1
from sklearn.base import clone


from mlens.utils.dummy import OLS
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from time import perf_counter

PLOT = True
if PLOT:
    try:
        import matplotlib.pyplot as plt
        plt.ion()

    except ImportError:
        print("Could not import matplotlib. Will ignore PLOT flag.")
        PLOT = False


ENS = [SuperLearner, BlendEnsemble]
KWG = [{'folds': 2, 'backend': 'threading'}, {}]
MAX = int(1e6)
STEP = int(1e5)
COLS = 100

SEED = 2017
np.random.seed(SEED)


def build_ensemble(kls, **kwargs):
    """Generate ensemble of class kls."""

    ens = kls(**kwargs)
    ens.add([OLS() for _ in range(20)])
    ens.add_meta(OLS())
    return ens

if __name__ == '__main__':
    X, y = make_friedman1(n_samples=int(1e6),
                          n_features=100, random_state=SEED)

    ens = SuperLearner(n_jobs=-1, folds=4, verbose=100)
    ens = LayerContainer(n_jobs=-1, verbose=100)
    ens.add([OLS() for _ in range(4)], 'stack',
            indexer=FoldIndex(2))

    t0 = perf_counter()
    ens.fit(X, y)
    t1 = perf_counter()

    print(t1 - t0)


if __name__ == 'sd':

    print("\nML-ENSEMBLE\n")
    print("Threading performance test for data set dimensioned up "
          "to (%i, %i)" % (MAX, COLS))

    c = os.cpu_count()
    print("Available CPUs: %i\n" % c)

    cores = [1, c]

    ens = [[build_ensemble(kls, n_jobs=i, **kwd)
            for kls, kwd in zip(ENS, KWG)]
           for i in cores]

    print('Ensemble architecture')
    print("Num layers: %i" % ens[0][0].layers.n_layers)
    print("Fit per base layer estimator: %i + 1" % ens[0][0].folds)

    for lyr in ens[0][0].layers.layers:
        if int(lyr[-1]) == ens[0][0].layers.n_layers:
            continue
        print('%s | Estimators: %r.' % (lyr, [e for e, _ in
                                              ens[0][0].layers.layers[
                                                  lyr].estimators]))
    print("%s | Meta Estimator: %s" % ('layer-2', ens[0][0].layers.layers[
        'layer-2'].estimators[0][0]))

    print('\nFIT TIMES')
    print('%7s' % 'samples', flush=True)

    ts = perf_counter()
    times = {i: {kls().__class__.__name__: []
             for kls in [SuperLearner, BlendEnsemble]}
             for i in cores}

    for s in range(STEP, MAX + STEP, STEP):
        X, y = make_friedman1(n_samples=s, n_features=COLS, random_state=SEED)

        for n, etypes in zip(cores, ens):
            print('%7i' % s, end=" ", flush=True)
            for e in etypes:
                name = e.__class__.__name__
                e = clone(e)

                t0 = perf_counter()
                e.fit(X, y)
                t1 = perf_counter() - t0

                times[n][name].append(t1)

                print('%s (%i) : %6.2fs |' % (name, n, t1),
                      end=" ", flush=True)
            print()
        print()

    print_time(ts, "Benchmark done")

    if PLOT:
        print("Plotting results...", end=" ", flush=True)
        x = range(STEP, MAX + STEP, STEP)

        cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0,
                                                     int(3 * len(cores)))
              ]
        plt.figure(figsize=(8, 8))

        i = 0
        for n in cores:
            for s, e in times[n].items():
                ax = plt.plot(x, e, color=cm[i], marker='.',
                              label='%s (%i)' % (s, n))
                i += 1

        plt.title('Benchmark of time to fit')
        plt.xlabel('Sample size')
        plt.ylabel('Time to fit (sec)')
        plt.legend(frameon=False)

        f = os.path.join('scale_comp_1.png')
        plt.savefig(f, bbox_inches='tight', dpi=600)
        print("done.\nFigure written to %s" % f)
