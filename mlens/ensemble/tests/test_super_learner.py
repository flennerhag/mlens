"""ML-ENSEMBLE

Place holder for more rigorous tests.

"""
import numpy as np
from mlens.metrics import rmse
from mlens.base import FoldIndex
from mlens.utils.dummy import Data, OLS, PREPROCESSING, ESTIMATORS, ECM

from mlens.ensemble import SuperLearner

import os
try:
    from contextlib import redirect_stdout
except ImportError:
    from mlens.externals.fixes import redirect as redirect_stdout

FOLDS = 3
LEN = 6
WIDTH = 2
MOD = 2

data1 = Data('stack', False, True, FOLDS)
X1, y1 = data1.get_data((LEN, WIDTH), MOD)
(F1, wf1), (P1, wp1) = data1.ground_truth(X1, y1, 1, False)
G1 = OLS().fit(F1, y1).predict(P1)

data2 = Data('stack', False, False, FOLDS)
X2, y2 = data1.get_data((LEN, WIDTH), MOD)
(F2, wf2), (P2, wp2) = data2.ground_truth(X2, y2, 1, False)
G2 = OLS().fit(F2, y2).predict(P2)

ens1 = SuperLearner(folds=FOLDS, scorer=rmse, verbose=100)
ens1.add(ESTIMATORS, PREPROCESSING, dtype=np.float64)
ens1.add_meta(OLS(), dtype=np.float64)

ens2 = SuperLearner(folds=FOLDS, scorer=rmse, verbose=100)
ens2.add(ECM, dtype=np.float64)
ens2.add_meta(OLS(), dtype=np.float64)


def test_run_w_folds():
    """[SuperLearner] 'fit' and 'predict' runs correctly with folds."""

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        ens1.fit(X1, y1)

        pred = ens1.predict(X1)

    np.testing.assert_array_equal(pred, G1)


def test_run_wo_folds():
    """[SuperLearner] 'fit' and 'predict' runs correctly without folds."""

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        ens2.fit(X2, y2)

        pred = ens2.predict(X2)

    np.testing.assert_array_equal(pred, G2)

def test_scores_w_folds():
    """[SuperLearner] test scoring with folds."""

    scores = {'no__null': [],
              'no__offs': [],
              'sc__offs': []}

    for _, tei in FoldIndex(FOLDS, X1).generate(as_array=True):
        col = 0
        for case in sorted(PREPROCESSING):
            for est_name, _ in ESTIMATORS[case]:
                s = rmse(y1[tei], F1[tei][:, col])
                scores['%s__%s' % (case, est_name)].append(s)

                col += 1

    for k in scores:
        scores[k] = np.mean(scores[k])

    for k in scores:

        assert scores[k] == ens1.scores_['score_mean'][('layer-1', k)]


def test_scores_wo_folds():
    """[SuperLearner] test scoring without folds."""

    scores = dict()
    for _, tei in FoldIndex(FOLDS, X2).generate(as_array=True):
        col = 0
        for est_name, _ in ECM:
            s = rmse(y2[tei], F2[tei][:, col])

            if not est_name in scores:
                scores[est_name] = []

            scores[est_name].append(s)

            col += 1

    for k in scores:
        scores[k] = np.mean(scores[k])

    for k in scores:

        assert scores[k] == ens2.scores_['score_mean'][('layer-1', k)]
