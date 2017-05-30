"""ML-ENSEMBLE

Comparison of multiprocessing performance as data scales.

Example Output
--------------

ML-ENSEMBLE

Threading performance test for data set dimensioned up to (10000, 50)
Available CPUs: 4

Ensemble architecture
Num layers: 2
Fit per base layer estimator: 2 + 1
layer-1 | Estimators: ['svr-1', 'svr-2', 'svr-3', 'svr-4'].
layer-2 | Meta Estimator: svr

FIT TIMES
samples
   1000 SuperLearner (1) :   0.88s | BlendEnsemble (1) :   0.35s |
   1000 SuperLearner (4) :   0.71s | BlendEnsemble (4) :   0.41s |

   2000 SuperLearner (1) :   2.82s | BlendEnsemble (1) :   0.76s |
   2000 SuperLearner (4) :   1.51s | BlendEnsemble (4) :   0.59s |

   3000 SuperLearner (1) :   6.04s | BlendEnsemble (1) :   1.56s |
   3000 SuperLearner (4) :   2.96s | BlendEnsemble (4) :   0.90s |

   4000 SuperLearner (1) :  10.94s | BlendEnsemble (1) :   2.79s |
   4000 SuperLearner (4) :   7.92s | BlendEnsemble (4) :   1.53s |

   5000 SuperLearner (1) :  18.45s | BlendEnsemble (1) :   4.58s |
   5000 SuperLearner (4) :   8.52s | BlendEnsemble (4) :   2.26s |

   6000 SuperLearner (1) :  27.48s | BlendEnsemble (1) :   7.24s |
   6000 SuperLearner (4) :  15.06s | BlendEnsemble (4) :   3.41s |

   7000 SuperLearner (1) :  38.73s | BlendEnsemble (1) :   8.62s |
   7000 SuperLearner (4) :  18.21s | BlendEnsemble (4) :   4.41s |

   8000 SuperLearner (1) :  52.08s | BlendEnsemble (1) :  12.10s |
   8000 SuperLearner (4) :  23.43s | BlendEnsemble (4) :   4.95s |

   9000 SuperLearner (1) :  61.70s | BlendEnsemble (1) :  14.58s |
   9000 SuperLearner (4) :  28.55s | BlendEnsemble (4) :   8.45s |

  10000 SuperLearner (1) :  75.76s | BlendEnsemble (1) :  18.72s |
  10000 SuperLearner (4) :  32.71s | BlendEnsemble (4) :   7.52s |

Benchmark done | 00:09:00

Plotting results... done.
Figure written to .../scale_comp_1.png
"""

import os
import numpy as np

from mlens.ensemble import SuperLearner, BlendEnsemble
from mlens.utils import print_time

from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.datasets import make_friedman1
from time import perf_counter

PLOT = True
ENS = [SuperLearner, BlendEnsemble]
KWG = [{'folds': 2}, {}]
MAX = int(1e4)
STEP = int(1e3)
COLS = 50

SEED = 2017
np.random.seed(SEED)


def build_ensemble(kls, **kwargs):
    """Generate ensemble of class kls."""

    ens = kls(**kwargs)
    ens.add([SVR() for _ in range(4)])
    ens.add_meta(SVR())
    return ens

if __name__ == '__main__':

    c = os.cpu_count()
    cores = [1, c]

    ens = [[build_ensemble(kls, n_jobs=i, **kwd)
            for kls, kwd in zip(ENS, KWG)]
           for i in cores]

    ###########################################################################
    # PRINTED MESSAGE
    print("\nML-ENSEMBLE\n")
    print("Threading performance test for data set "
          "dimensioned up to (%i, %i)" % (MAX, COLS))
    print("Available CPUs: %i\n" % c)
    print('Ensemble architecture')
    print("Num layers: %i" % ens[0][0].layers.n_layers)
    print("Fit per base layer estimator: %i + 1" % ens[0][0].folds)

    for lyr in ens[0][0].layers.layers:
        if int(lyr[-1]) == ens[0][0].layers.n_layers:
            continue

        print('%s | Estimators: %r.' %
              (lyr, [e for e, _ in ens[0][0].layers.layers[lyr].estimators]))

    print("%s | Meta Estimator: %s" %
          ('layer-2', ens[0][0].layers.layers['layer-2'].estimators[0][0]))

    print('\nFIT TIMES')
    print('%7s' % 'samples', flush=True)

    ###########################################################################
    # ESTIMATION
    times = {i: {kls().__class__.__name__: []
             for kls in [SuperLearner, BlendEnsemble]}
             for i in cores}

    ts = perf_counter()
    for s in range(STEP, MAX + STEP, STEP):

        X, y = make_friedman1(n_samples=s, n_features=COLS, random_state=SEED)

        # Iterate over number of cores to fit with
        for n, etypes in zip(cores, ens):

            print('%7i' % s, end=" ", flush=True)

            # Iterate over ensembles with given number of cores
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
        try:
            import matplotlib.pyplot as plt

            plt.ion()
            print("Plotting results...", end=" ", flush=True)

            plt.figure(figsize=(8, 8))

            x = range(STEP, MAX + STEP, STEP)
            cm = [plt.cm.rainbow(i)
                  for i in np.linspace(0, 1.0, int(3 * len(cores)))]

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

            f = os.path.join(os.getcwd(), 'scale_cpu.png')
            plt.savefig(f, bbox_inches='tight', dpi=600)
            print("done.\nFigure written to %s" % f)

        except ImportError:
            print("Could not import matplotlib. Will ignore PLOT flag.")
