"""ML-ENSEMBLE

Comparison of multiprocessing performance as data scales.

Example Output
--------------

ML-ENSEMBLE

Ensemble scale benchmark for datadimensioned up to (250000, 20)
Available CPUs: 4

Ensemble architecture
Num layers: 2
layer-1 | Estimators: ['svr', 'randomforestregressor', 'gradientboostingregressor', 'lasso', 'mlpregressor'].
layer-2 | Meta Estimator: lasso

SCORES (TIME TO FIT)
Sample size
      20000 SuperLearner : 0.807 ( 19.83s) | BlendEnsemble : 0.823 (  4.09s) | Subsemble : 0.789 (  9.84s) |
      40000 SuperLearner : 0.396 ( 42.94s) | BlendEnsemble : 0.462 ( 11.37s) | Subsemble : 0.777 ( 28.49s) |
      60000 SuperLearner : 0.280 ( 75.08s) | BlendEnsemble : 0.328 ( 23.43s) | Subsemble : 0.570 ( 56.93s) |
      80000 SuperLearner : 0.310 (126.59s) | BlendEnsemble : 0.414 ( 41.75s) | Subsemble : 0.434 ( 90.66s) |
     100000 SuperLearner : 0.447 (180.77s) | BlendEnsemble : 0.308 ( 63.80s) | Subsemble : 0.541 (111.31s) |
     120000 SuperLearner : 0.306 (243.34s) | BlendEnsemble : 0.281 ( 92.71s) | Subsemble : 0.323 (129.15s) |
     140000 SuperLearner : 0.269 (312.58s) | BlendEnsemble : 0.408 (107.19s) | Subsemble : 0.303 (165.86s) |
     160000 SuperLearner : 0.298 (410.33s) | BlendEnsemble : 0.312 (145.76s) | Subsemble : 0.343 (234.12s) |
     180000 SuperLearner : 0.250 (614.27s) | BlendEnsemble : 0.279 (195.74s) | Subsemble : 0.272 (295.76s) |
     200000 SuperLearner : 0.301 (594.41s) | BlendEnsemble : 0.390 (208.11s) | Subsemble : 0.260 (265.42s) |
     220000 SuperLearner : 0.280 (787.79s) | BlendEnsemble : 0.260 (251.45s) | Subsemble : 0.407 (356.17s) |
     240000 SuperLearner : 0.304 (928.15s) | BlendEnsemble : 0.299 (314.76s) | Subsemble : 0.300 (459.59s) |
     260000 SuperLearner : 0.252 (1226.66s) | BlendEnsemble : 0.273 (350.77s) | Subsemble : 0.279 (462.97s) |
Benchmark done | 04:20:34

Plotting results...
Figure written to /Users/Sebastian/Documents/python/mlens_dev/scale_benchmark2_time.png
Figure written to /Users/Sebastian/Documents/python/mlens_dev/scale_benchmark2_score.png
done.

"""

import os
import numpy as np

from mlens.ensemble import SuperLearner, BlendEnsemble, Subsemble
from mlens.utils import print_time
from mlens.metrics import rmse

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.datasets import make_friedman1
from time import perf_counter
import warnings

PLOT = True
ENS = [SuperLearner, BlendEnsemble, Subsemble]
KWG = [{'folds': 2}, {}, {'partitions': 3, 'folds': 2}]
MAX = int(2.5 * 1e5)
STEP = int(2*1e4)
COLS = 20

SEED = 2017
np.random.seed(SEED)


def build_ensemble(kls, **kwargs):
    """Generate ensemble of class kls."""

    ens = kls(**kwargs)
    ens.add([SVR(), RandomForestRegressor(),
             GradientBoostingRegressor(), Lasso(copy_X=False),
             MLPRegressor(shuffle=False, alpha=0.001)])
    ens.add_meta(Lasso(copy_X=False))
    return ens

if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    c = os.cpu_count()

    ens = [build_ensemble(kls, n_jobs=-1, **kwd) for kls, kwd in zip(ENS, KWG)]

    ###########################################################################
    # PRINTED MESSAGE
    print("\nML-ENSEMBLE\n")
    print("Ensemble scale benchmark for data"
          "dimensioned up to (%i, %i)" % (MAX, COLS))
    print("Available CPUs: %i\n" % c)
    print('Ensemble architecture')
    print("Num layers: %i" % ens[0].layers.n_layers)

    for lyr in ens[0].layers.layers:
        if int(lyr[-1]) == ens[0].layers.n_layers:
            continue

        print('%s | Estimators: %r.' %
              (lyr, [e for e, _ in ens[0].layers.layers[lyr].estimators]))

    print("%s | Meta Estimator: %s" %
          ('layer-2', ens[0].layers.layers['layer-2'].estimators[0][0]))

    print('\nSCORES (TIME TO FIT)')
    print('%11s' % 'Sample size', flush=True)

    ###########################################################################
    # ESTIMATION
    times = {kls().__class__.__name__: [] for kls in ENS}
    scores = {kls().__class__.__name__: [] for kls in ENS}

    ts = perf_counter()
    for s in range(STEP, MAX + STEP, STEP):

        q = int(np.floor(s / 2))

        print('%11i' % s, end=" ", flush=True)

        X, y = make_friedman1(n_samples=s, n_features=COLS, random_state=SEED)

        # Iterate over ensembles with given number of cores
        for e in ens:
            name = e.__class__.__name__
            e = clone(e)

            t0 = perf_counter()
            e.fit(X[:q], y[:q])
            t1 = perf_counter() - t0

            sc = rmse(y[q:], e.predict(X[q:]))

            times[name].append(t1)
            scores[name].append(sc)

            print('%s : %.3f (%6.2fs) |' % (name, sc, t1), end=" ", flush=True)
        print()

    print_time(ts, "Benchmark done")

    if PLOT:
        try:
            import matplotlib.pyplot as plt

            plt.ion()
            print("Plotting results...", flush=True)

            plt.figure(figsize=(8, 8))

            x = range(STEP, MAX + STEP, STEP)
            cm = [plt.cm.rainbow(i)
                  for i in np.linspace(0, 1.0, int(len(ENS)))]

            for i, (s, e) in enumerate(times.items()):
                ax = plt.plot(x, e, color=cm[i], marker='.', label='%s' % s)

            plt.xticks([i for i in range(STEP, MAX + STEP, 2*STEP)],
                       [int(i / 2) for i in range(STEP, MAX + STEP, 2*STEP)])
            plt.title('Time to fit')
            plt.xlabel('Sample size')
            plt.ylabel('Time to fit (sec)')
            plt.legend(frameon=False)

            f = os.path.join(os.getcwd(), 'scale_ens_time.png')
            plt.savefig(f, bbox_inches='tight', dpi=600)
            print("Figure written to %s" % f)

            plt.figure(figsize=(8, 8))

            x = range(STEP, MAX + STEP, STEP)
            cm = [plt.cm.rainbow(i)
                  for i in np.linspace(0, 1.0, int(len(ENS)))]

            for i, (s, e) in enumerate(scores.items()):
                ax = plt.plot(x, e, color=cm[i], marker='.', label='%s' % s)

            plt.xticks([i for i in range(STEP, MAX + STEP, 2*STEP)],
                       [int(i / 2) for i in range(STEP, MAX + STEP, 2*STEP)])
            plt.title('Test set accuracy')
            plt.xlabel('Sample size')
            plt.ylabel('Root Mean Square Error')
            plt.legend(frameon=False)

            f = os.path.join(os.getcwd(), 'docs/img/scale_ens_score.png')
            plt.savefig(f, bbox_inches='tight', dpi=600)
            print("Figure written to %s" % f)

            print("done.")
        except ImportError:
            print("Could not import matplotlib. Will ignore PLOT flag.")
