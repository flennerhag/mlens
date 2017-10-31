"""ML-ENSEMBLE
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

FOLDS = 3
LEN = 24
WIDTH = 2
MOD = 2

data = Data('stack', False, True, FOLDS)
X, y = data.get_data((LEN, WIDTH), MOD)

est = EstimatorContainer()
lc_s = est.get_layer_estimator('stack', False, True)
lc_b = est.get_layer_estimator('blend', False, False)
lc_u = est.get_layer_estimator('subsemble', False, False)

l_s = est.get_layer('stack', False, True)
l_b = est.get_layer('blend', False, False)
l_u = est.get_layer('subsemble', False, False)

seq = Sequential(stack=[l_s, l_b, l_u])


def test_fit_seq():
    """[Sequential] Test multilayer fitting."""
    S = lc_s.fit_transform(X, y)
    B = lc_b.fit_transform(S, y)
    r = y.shape[0] - B.shape[0]
    U = lc_u.fit_transform(B, y[r:])
    out = seq.fit_transform(X, y)
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
    S = lc_s.fit_transform(X, y)
    B = lc_b.fit_transform(S, y)
    r = y.shape[0] - B.shape[0]
    U = lc_u.fit_transform(B, y[r:])

    ens = SequentialEnsemble()
    ens.add('stack', ESTIMATORS, PREPROCESSING, dtype=np.float64)
    ens.add('blend', ECM, dtype=np.float64)
    ens.add('subsemble', ECM, dtype=np.float64)

    out = ens.fit_transform(X, y)
    np.testing.assert_array_equal(U, out)


def test_predict():
    """[SequentialEnsemble] Test multilayer prediction."""
    S = lc_s.predict(X)
    B = lc_b.predict(S)
    U = lc_u.predict(B)
    ens = SequentialEnsemble()
    ens.add('stack', ESTIMATORS, PREPROCESSING, dtype=np.float64)
    ens.add('blend', ECM, dtype=np.float64)
    ens.add('subsemble', ECM, dtype=np.float64)
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
    seq.add('subsemble', ECM, dtype=np.float64)

    F = ens.fit(X, y).predict(X)
    P = seq.fit(X, y).predict(X)

    np.testing.assert_array_equal(P, F)
