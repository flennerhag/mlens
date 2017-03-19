"""ML-ENSMEMBLE

Memory profiling of mlens against Scikit-learn estimators.
"""
import numpy as np

from mlens.utils import print_time
from mlens.ensemble import StackingEnsemble

from sklearn.datasets import make_friedman1

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
import time


MAX = 100000
SEED = 100
SLEEP = MAX / 10000 if MAX <= 100000 else MAX / 20000

np.random.seed(SEED)


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

    ens.add([GradientBoostingRegressor()])

    return ens


@profile
def fit_ens():
    """Fit ensemble."""
    print("Fitting ensemble...", end=" ")
    ens = build_ensemble(shuffle=False)
    ens.fit(X, y)
    print("Done.")


@profile
def fit_gbm():
    """Fit gbm."""
    print("Fitting GBM...", end=" ")
    gbm = GradientBoostingRegressor()
    gbm.fit(X, y)
    print("Done.")


@profile
def fit_rf():
    """Fit Random Forest."""
    print("Fitting Random Forest...", end=" ")
    rf = RandomForestRegressor(random_state=SEED)
    rf.fit(X, y)
    print("Done.")


@profile
def fit_KNN():
    """Fit KNN."""
    print("Fitting KNN...", end=" ")
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor())
    knn.fit(X, y)
    print("Done.")


@profile
def fit_las():
    """Fit Lasso."""
    print("Fitting Lasso...", end=" ")
    ls = make_pipeline(StandardScaler(), Lasso())
    ls.fit(X, y)
    print("Done.")


@profile
def fit_svr():
    """Fit SVR."""
    print("Fitting Lasso...", end=" ")
    svr = make_pipeline(MinMaxScaler(), SVR())
    svr.fit(X, y)
    print("Done.")


if __name__ == '__main__':

    X, y = make_friedman1(MAX)
    print("Profiling memory with dense data.\n"
          "shape: (%i, %i)\n"
          "size: %i MB\n" % (MAX, 10, np.ceil(X.nbytes / 1e+6)))

    t0 = time.time()

    time.sleep(SLEEP)
    fit_ens()

    time.sleep(SLEEP)
    fit_gbm()

    time.sleep(SLEEP)
    fit_rf()

    time.sleep(SLEEP)
    fit_KNN()

    time.sleep(SLEEP)
    fit_las()

    time.sleep(SLEEP)
    fit_svr()

    time.sleep(SLEEP)
    print_time(t0, "Profiling complete.")
