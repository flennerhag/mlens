"""ML-ENSEMBLE

Place holder for more rigorous tests.

"""
import numpy as np
from mlens.metrics import rmse
from mlens.base import FoldIndex
from mlens.utils.dummy import Data, OLS, PREPROCESSING, ESTIMATORS

from mlens.ensemble import SuperLearner


FOLDS = 3
LEN = 6
WIDTH = 2
MOD = 2

data = Data('stack', False, True, FOLDS)
X, y = data.get_data((LEN, WIDTH), MOD)

(F, wf), (P, wp) = data.ground_truth(X, y, 1, False)


def test_run():
    """[SuperLearner] 'fit' and 'predict' runs correctly."""
    meta = OLS()
    meta.fit(F, y)
    g = meta.predict(P)

    ens = SuperLearner(folds=FOLDS, scorer=rmse)
    ens.add(ESTIMATORS, PREPROCESSING)
    ens.add_meta(OLS())

    ens.fit(X, y)

    pred = ens.predict(X)

    np.testing.assert_array_equal(pred, g)


def test_scores():
    """[SuperLearner] test scoring."""
    meta = OLS()
    meta.fit(F, y)
    g = meta.predict(P)

    ens = SuperLearner(folds=FOLDS, scorer=rmse)
    ens.add(ESTIMATORS, PREPROCESSING)
    ens.add_meta(OLS())

    ens.fit(X, y)

    pred = ens.predict(X)

    scores = {'no__null': [],
              'no__offs': [],
              'sc__offs': []}

    for _, tei in FoldIndex(FOLDS, X).generate(as_array=True):
        col = 0
        for case in sorted(PREPROCESSING):
            for est_name, _ in ESTIMATORS[case]:
                s = rmse(y[tei], F[tei][:, col])
                scores['%s__%s' % (case, est_name)].append(s)

                col += 1

    for k in scores:
        scores[k] = np.mean(scores[k])

    for k in scores:

        assert scores[k] == ens.scores_['score_mean'][('layer-1', k)]
