"""ML-ENSMEMBLE

Memory profiling of mlens against Scikit-learn estimators.

# Run from command line using a memory profiler that supports memory
consumption comparison over time.

Examples
--------

Using mprof:

>>> mprof run friedman_memory.py
mprof: Sampling memory every 0.1s
running as a Python program...

ML-ENSEMBLE

Benchmark of ML-ENSEMBLE memory profile against Scikit-learn estimators.

Data shape: (1000000, 50)

Data size: 400 MB

Fitting LAS... Done | 00:00:01

Fitting KNN... Done | 00:00:08

Fitting ENS... Done | 00:00:21

Fitting ELN... Done | 00:00:01

Profiling complete. | 00:01:13


>>> mprof plot friedman_memory.py -t "Memory Consumption Benchmark"
Using last profile data.

.. image:: memory.png
"""

import numpy as np

from mlens.utils import print_time
from mlens.ensemble import SuperLearner

from sklearn.datasets import make_friedman1

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
import time


MAX = int(1e6)
COLS = 50
SEED = 2017

SLEEP = 10

np.random.seed(SEED)


def build_ensemble(**kwargs):
    """Generate ensemble."""

    ens = SuperLearner(**kwargs)

    est = [ElasticNet(copy_X=False),
           Lasso(copy_X=False)]

    ens.add(est)
    ens.add(KNeighborsRegressor())

    return ens


@profile
def ensemble():
    """Fit ensemble."""
    print("Fitting ENS...", end=" ", flush=True)
    time.sleep(SLEEP)
    t0 = time.time()
    ens = build_ensemble(shuffle=False, folds=2)
    ens.fit(X, y)
    print_time(t0, "Done", end="")


@profile
def knn():
    """Fit KNN."""
    print("Fitting KNN...", end=" ", flush=True)
    time.sleep(SLEEP)
    t0 = time.time()
    knn = KNeighborsRegressor()
    knn.fit(X, y)
    print_time(t0, "Done", end="")


@profile
def lasso():
    """Fit Lasso."""
    print("Fitting LAS...", end=" ", flush=True)
    time.sleep(SLEEP)
    t0 = time.time()
    ls = Lasso()
    ls.fit(X, y)
    print_time(t0, "Done", end="")


@profile
def elasticnet():
    """Fit Elastic Net."""
    print("Fitting ELN...", end=" ", flush=True)
    time.sleep(SLEEP)
    t0 = time.time()
    ls = Lasso()
    ls.fit(X, y)
    print_time(t0, "Done", end="")


if __name__ == '__main__':

    X, y = make_friedman1(MAX, COLS)

    print("\nML-ENSEMBLE\n")
    print("Benchmark of ML-ENSEMBLE memory profile against "
          "Scikit-learn estimators.\n"
          "Data shape: (%i, %i)\n"
          "Data size: %i MB\n" % (MAX, COLS, np.ceil(X.nbytes / 1e+6)))

    ts = time.time()

    lasso()
    knn()
    ensemble()
    elasticnet()

    print_time(ts, "\nProfiling complete.")
