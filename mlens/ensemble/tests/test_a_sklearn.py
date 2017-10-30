"""
Test Scikit-learn
"""
import numpy as np
from mlens.ensemble import SuperLearner, Subsemble, BlendEnsemble
try:
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures

    from sklearn.datasets import load_boston

    has_sklearn = True

except ImportError:
    has_sklearn = False


if has_sklearn:

    X, y = load_boston(True)

    estimators = [Lasso(),
                  GradientBoostingRegressor(),
                  LinearRegression(),
                  KNeighborsRegressor(),
                  SVR(),
                  RandomForestRegressor(),
                  ]

    est_prep = {'prep1': estimators,
                'prep2': estimators,
                'prep3': estimators}

    prep_1 = [PCA()]
    prep_2 = [PolynomialFeatures(), StandardScaler()]

    prep = {'prep1': prep_1,
            'prep2': prep_2,
            'prep3': []}

    def get_ensemble(cls, backend, preprocessing):
        """Get ensemble."""
        if preprocessing:
            est = est_prep
        else:
            est = estimators
        ens = cls(backend=backend)
        ens.add(est, preprocessing)
        ens.add(LinearRegression(), meta=True)
        return ens

    def test_super_learner_s_m():
        """[SuperLearner] Test scikit-learn comp - mp | np"""
        ens = get_ensemble(SuperLearner, 'multiprocessing', None)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype


    def test_super_learner_f_m():
        """[SuperLearner] Test scikit-learn comp - mp | p"""
        ens = get_ensemble(SuperLearner, 'multiprocessing', prep)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

    def test_super_learner_s_t():
        """[SuperLearner] Test scikit-learn comp - th | np"""
        ens = get_ensemble(SuperLearner, 'threading', None)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

    def test_super_learner_f_t():
        """[SuperLearner] Test scikit-learn comp - th | p"""
        ens = get_ensemble(SuperLearner, 'threading', prep)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

    def test_subsemble_s_m():
        """[Subsemble] Test scikit-learn comp - mp | np"""
        ens = get_ensemble(Subsemble, 'multiprocessing', None)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype


    def test_subsemble_f_m():
        """[Subsemble] Test scikit-learn comp - mp | p"""
        ens = get_ensemble(Subsemble, 'multiprocessing', prep)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

    def test_subsemble_s_t():
        """[Subsemble] Test scikit-learn comp - th | np"""
        ens = get_ensemble(Subsemble, 'threading', None)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

    def test_subsemble_f_t():
        """[Subsemble] Test scikit-learn comp - th | p"""
        ens = get_ensemble(Subsemble, 'threading', prep)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

    def test_blend_s_m():
        """[BlendEnsemble] Test scikit-learn comp - mp | np"""
        ens = get_ensemble(BlendEnsemble, 'multiprocessing', None)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype


    def test_blend_f_m():
        """[BlendEnsemble] Test scikit-learn comp - mp | p"""
        ens = get_ensemble(BlendEnsemble, 'multiprocessing', prep)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

    def test_blend_s_m():
        """[BlendEnsemble] Test scikit-learn comp - th | np"""
        ens = get_ensemble(BlendEnsemble, 'threading', None)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

    def test_blend_f_m():
        """[BlendEnsemble] Test scikit-learn comp - th | p"""
        ens = get_ensemble(BlendEnsemble, 'threading', prep)
        ens.fit(X, y)
        p = ens.predict(X)
        assert p.shape == y.shape
        assert p.dtype == ens.layer_1.dtype

