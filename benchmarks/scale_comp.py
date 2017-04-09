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
from mlens.ensemble import SuperLearner, Subsemble, BlendEnsemble
from mlens.utils import print_time

from sklearn.datasets import make_friedman1
from sklearn.base import clone


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

MAX = int(50000)
STEP = int(5000)
COLS = 5

SEED = 2017
np.random.seed(SEED)


def build_ensemble(kls, **kwargs):
    """Generate ensemble of class kls."""

    ens = kls(**kwargs)

    ens.add([KNeighborsRegressor(), KNeighborsRegressor(),
             KNeighborsRegressor(), KNeighborsRegressor()])

    ens.add(KNeighborsRegressor())

    return ens


if __name__ == '__main__':

    print("\nML-ENSEMBLE\n")
    print("Threading performance test for data set dimensioned up "
          "to (%i, %i)" % (MAX, COLS))

    c = os.cpu_count()
    print("Available CPUs: %i\n" % c)

    cores = [int(np.floor(i)) for i in np.linspace(1, c, 3)]

    ens = [[build_ensemble(kls, n_jobs=i, )
            for kls in [SuperLearner, Subsemble, BlendEnsemble]]
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
             for kls in [SuperLearner, Subsemble, BlendEnsemble]}
             for i in cores}

    for s in range(STEP, MAX + STEP, STEP):
        X, y = make_friedman1(n_samples=s, random_state=SEED)

        for n, etypes in zip(cores, ens):
            print('%7i' % s, end=" ", flush=True)
            for e in etypes:
                name = e.__class__.__name__
                e = clone(e)

                t0 = perf_counter()
                e.fit(X, y)
                t1 = perf_counter() - t0

                times[n][name].append(t1)

                print('%s (%i) : %4.2f |' % (name, n, t1), end=" ", flush=True)
            print()
        print()

    print_time(ts, "Benchmark done")

    if PLOT:
        print("Plotting results...", end=" ", flush=True)
        x = range(STEP, MAX + STEP, STEP)

        cm = [plt.cm.rainbow(i) for i in np.linspace(0, 1.0,
                                                     int(3 *len(cores)))
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

        f = os.path.join('scale_comp.png')
        plt.savefig(f, bbox_inches='tight', dpi=600)
        print("done.\nFigure written to %s" % f)
