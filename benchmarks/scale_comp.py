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
layer-1 | Estimators: ['RandomForestRegressor', 'GradientBoostingRegressor', 'ElasticNet', 'KNeighborsRegressor'].
layer-2 | Meta Estimator: Lasso

FIT TIMES
samples | cores: 1 | cores: 2 | cores: 4 |


"""

import numpy as np

import os
from mlens.ensemble import StackingEnsemble
from mlens.utils import print_time

from sklearn.datasets import make_friedman1
from sklearn.base import clone

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

MAX = int(1e6)
STEP = int(1e5)
COLS = 10

SEED = 2017
SLEEP = 10
np.random.seed(SEED)


def build_ensemble(**kwargs):
    """Generate ensemble."""

    ens = StackingEnsemble(**kwargs)

    ens.add([ElasticNet(copy_X=False),
             RandomForestRegressor(),
             Lasso(),
             KNeighborsRegressor()])

    ens.add(Lasso())

    return ens


if __name__ == '__main__':

    print("\nML-ENSEMBLE\n")
    print("Threading performance test for data set dimensioned up "
          "to (%i, %i)" % (MAX, COLS))

    c = os.cpu_count()
    print("Available CPUs: %i\n" % c)

    cores = [int(np.floor(i)) for i in np.linspace(1, c, 3)]

    ens = [build_ensemble(n_jobs=i, folds=4, shuffle=False) for i in cores]

    print('Ensemble architecture')
    print("Num layers: %i" % ens[0].layers.n_layers)
    print("Fit per base layer estimator: %i + 1" % ens[0].folds)

    for lyr in ens[0].layers.layers:
        if int(lyr[-1]) == ens[0].layers.n_layers:
            continue
        print('%s | Estimators: %r.' % (lyr, [e for e, _ in
                                              ens[0].layers.layers[
                                                  lyr].estimators]))
    print("%s | Meta Estimator: %s" % ('layer-2', ens[0].layers.layers[
        'layer-2'].estimators[0][0]))

    print('\nFIT TIMES')
    print('%7s' % 'samples', end=' | ')

    for n in cores:
        print('cores: %s' % n, end=' | ')
    print()

    ts = perf_counter()
    times = {i: [] for i in cores}
    for s in range(STEP, MAX + STEP, STEP):

        print('%7i' % s, end=' | ', flush=True)
        X, y = make_friedman1(n_samples=s, random_state=SEED)

        for n, e in zip(cores, ens):
            e = clone(e)
            t0 = perf_counter()
            e.fit(X, y)
            t1 = perf_counter() - t0
            times[n].append(t1)

            m, s = divmod(t1, 60)
            print(' %02d:%02d  ' % (m, s), end=' | ', flush=True)
        print()

    print_time(ts, "Benchmark done")

    if PLOT:
        print("Plotting results...", end=" ", flush=True)
        x = range(STEP, MAX + STEP, STEP)

        cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0, len(cores))]
        plt.figure(figsize=(8, 8))

        for i, n in enumerate(cores):
            ax = plt.plot(x, times[n], color=cm[i], marker='.',
                          label='cores: %i' % n)

        plt.title('Benchmark of time to fit')
        plt.xlabel('Sample size')
        plt.ylabel('Time to fit (sec)')
        plt.legend(frameon=False)

        f = os.path.join('cale_comp.png')
        plt.savefig(f, bbox_inches='tight', dpi=600)
        print("done.\nFigure written to %s" % f)
