"""
=======================
MNIST dataset benchmark
=======================

Benchmark on the MNIST dataset.  The dataset comprises 70,000 samples
and 784 features. Here, we consider the task of predicting
10 classes -  digits from 0 to 9 from their raw images. By contrast to the
covertype dataset, the feature space is homogenous.

Example of output :
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
"""
from __future__ import division, print_function

# Author: Issam H. Laradji
#         Arnaud Joly <arnaud.v.joly@gmail.com>
# License: BSD 3 clause

# Adapted by: Sebastian Flennerhag

import os
from time import time
import argparse
import numpy as np

from mlens.utils import safe_print
from mlens.ensemble import Subsemble

from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_mldata
from sklearn.datasets import get_data_home
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.externals.joblib import Memory
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import zero_one_loss
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), 'mnist_benchmark_data'),
                mmap_mode='r')


@memory.cache
def load_data(dtype=np.float32, order='F'):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    safe_print("Loading dataset...")
    data = fetch_mldata('MNIST original')
    X = check_array(data['data'], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    safe_print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


ESTIMATORS = {
    "dummy": DummyClassifier(),
    'CART': DecisionTreeClassifier(),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=0),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=0),
    'Nystroem-SVM': make_pipeline(
        Nystroem(gamma=0.015, n_components=1000), LinearSVC(C=100)),
    'SampledRBF-SVM': make_pipeline(
        RBFSampler(gamma=0.015, n_components=1000), LinearSVC(C=100)),
    'LogisticRegression-SAG': LogisticRegression(solver='sag', tol=1e-1,
                                                 C=1e4),
    'LogisticRegression-SAGA': LogisticRegression(solver='saga', tol=1e-1,
                                                  C=1e4),
    'MultilayerPerceptron': MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
        solver='sgd', learning_rate_init=0.2, momentum=0.9, verbose=1,
        tol=1e-4, random_state=1),
    'MLP-adam': MLPClassifier(
        hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
        solver='adam', learning_rate_init=0.001, verbose=1,
        tol=1e-4, random_state=0)
}


def build_ensemble(cls, **kwargs):
    """Build ML-Ensemble"""
    ens = cls(**kwargs)

    use = ["ExtraTrees", "RandomForest",
           "LogisticRegression-SAG", "MLP-adam"]

    meta = RandomForestClassifier(n_estimators=100,
                                  random_state=0,
                                  n_jobs=-1)
    base_learners = list()
    for est_name, est in ESTIMATORS.items():
        e = clone(est)
        if est_name not in use:
            continue
        elif est_name == "MLP-adam":
            e.verbose = False
        try:
            e.set_params(**{'n_jobs': 1})
        except ValueError:
            pass

        base_learners.append((est_name, e))
    ens.add(base_learners, proba=True, shuffle=True, random_state=1)
    ens.add_meta(meta, shuffle=True, random_state=2)
    return ens


if __name__ == "__main__":
    ESTIMATORS['Subsemble'] = build_ensemble(
        Subsemble,
        partition_estimator=MiniBatchKMeans(n_clusters=5, random_state=0),
        partitions=5, verbose=1, folds=2, n_jobs=-2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--classifiers', nargs="+",
                        choices=ESTIMATORS, type=str,
                        default=['Subsemble', 'BlendEnsemble'],
                        help="list of classifiers to benchmark.")
    parser.add_argument('--order', nargs="?", default="C", type=str,
                        choices=["F", "C"],
                        help="Allow to choose between fortran and C ordered "
                             "data")
    args = vars(parser.parse_args())

    safe_print(__doc__)

    X_train, X_test, y_train, y_test = load_data(order=args["order"])

    safe_print("")
    safe_print("Dataset statistics:")
    safe_print("===================")
    safe_print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    safe_print("%s %d" % ("number of classes:".ljust(25),
                          np.unique(y_train).size))
    safe_print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    safe_print("%s %d (size=%dMB)" % ("number of train samples:".ljust(25),
                                      X_train.shape[0],
                                      int(X_train.nbytes / 1e6)))
    safe_print("%s %d (size=%dMB)" % ("number of test samples:".ljust(25),
                                      X_test.shape[0],
                                      int(X_test.nbytes / 1e6)))
    safe_print()
    safe_print("Training Classifiers")
    safe_print("====================")
    error, train_time, test_time = {}, {}, {}
    for name in sorted(args["classifiers"]):
        estimator = ESTIMATORS[name]
        safe_print("Training %s ... " % name)

        time_start = time()
        estimator.fit(X_train, y_train)
        train_time[name] = time() - time_start
        time_start = time()
        pre = estimator.predict(X_test)
        test_time[name] = time() - time_start
        error[name] = zero_one_loss(y_test, pre)
        safe_print("done")

    safe_print()
    safe_print("Classification performance:")
    safe_print("===========================")
    safe_print("{0: <24} {1: >10} {2: >11} {3: >12}"
               "".format("Classifier  ",
                         "train-time",
                         "test-time",
                         "error-rate"))

    safe_print("-" * 60)
    for name in sorted(args["classifiers"], key=error.get):
        safe_print("{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}"
                   "".format(name,
                             train_time[name],
                             test_time[name],
                             error[name]))
    safe_print()
