"""ML-ENSEMBLE

Place holder for more rigorous tests.

"""
import numpy as np
from mlens.ensemble import (SequentialEnsemble,
                            SuperLearner,
                            BlendEnsemble,
                            Subsemble)

from mlens.ensemble.base import Sequential
from mlens.testing.dummy import (Data,
                                 PREPROCESSING,
                                 ESTIMATORS,
                                 ECM,
                                 EstimatorContainer)

from mlens.externals.sklearn.base import clone

FOLDS = 3
LEN = 24
WIDTH = 2
MOD = 2

data = Data('stack', False, True, FOLDS)
X, y = data.get_data((LEN, WIDTH), MOD)

est = EstimatorContainer()
lc_s = est.get_layer('stack', False, True)
lc_b = est.get_layer('blend', False, False)
lc_u = est.get_layer('subset', False, False)

a = clone(lc_s)
a.name += '-1'
b = clone(lc_b)
b.name += '-2'
c = clone(lc_u)
c.name += '-3'
seq = Sequential()(a, b, c)

lc_s = Sequential(lc_s)
lc_b = Sequential(lc_b)
lc_u = Sequential(lc_u)


def test_fit_seq():
    """[Sequential] Test multilayer fitting."""
    S = lc_s.fit(X, y, return_preds=True)
    B = lc_b.fit(S, y, return_preds=True)
    r = y.shape[0] - B.shape[0]
    U = lc_u.fit(B, y[r:], return_preds=True)

    out = seq.fit(X, y, return_preds=True)
    np.testing.assert_array_equal(U, out)


def test_predict_seq():
    """[Sequential] Test multilayer prediction."""
    S = lc_s.predict(X)
    B = lc_b.predict(S)
    U = lc_u.predict(B)
    out = seq.predict(X)
    np.testing.assert_array_equal(U, out)


def test_fit():
    """[SequentialEnsemble] Test multilayer fitting."""
    S = lc_s.fit(X, y, return_preds=True)
    B = lc_b.fit(S, y, return_preds=True)
    r = y.shape[0] - B.shape[0]
    U = lc_u.fit(B, y[r:], return_preds=True)

    ens = SequentialEnsemble()
    ens.add('stack', ESTIMATORS, PREPROCESSING, dtype=np.float64)
    ens.add('blend', ECM, dtype=np.float64)
    ens.add('subset', ECM, dtype=np.float64)

    out = ens.fit(X, y, return_preds=True)
    np.testing.assert_array_equal(U, out)


def test_predict():
    """[SequentialEnsemble] Test multilayer prediction."""
    S = lc_s.predict(X)
    B = lc_b.predict(S)
    U = lc_u.predict(B)
    ens = SequentialEnsemble()
    ens.add('stack', ESTIMATORS, PREPROCESSING, dtype=np.float64)
    ens.add('blend', ECM, dtype=np.float64)
    ens.add('subset', ECM, dtype=np.float64)
    out = ens.fit(X, y).predict(X)
    np.testing.assert_array_equal(U, out)


def test_equivalence_super_learner():
    """[SequentialEnsemble] Test ensemble equivalence with SuperLearner."""

    ens = SuperLearner()
    seq = SequentialEnsemble()

    ens.add(ECM, dtype=np.float64)
    seq.add('stack', ECM, dtype=np.float64)

    F = ens.fit(X, y).predict(X)
    P = seq.fit(X, y).predict(X)

    np.testing.assert_array_equal(P, F)


def test_equivalence_blend():
    """[SequentialEnsemble] Test ensemble equivalence with BlendEnsemble."""

    ens = BlendEnsemble()
    seq = SequentialEnsemble()

    ens.add(ECM, dtype=np.float64)
    seq.add('blend', ECM, dtype=np.float64)

    F = ens.fit(X, y).predict(X)
    P = seq.fit(X, y).predict(X)

    np.testing.assert_array_equal(P, F)


def test_equivalence_subsemble():
    """[SequentialEnsemble] Test ensemble equivalence with Subsemble."""

    ens = Subsemble(n_jobs=1)
    seq = SequentialEnsemble(n_jobs=1)

    ens.add(ECM, dtype=np.float64)
    seq.add('subset', ECM, dtype=np.float64)

    F = ens.fit(X, y).predict(X)
    P = seq.fit(X, y).predict(X)

    np.testing.assert_array_equal(P, F)
