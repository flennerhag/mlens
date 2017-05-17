"""ML-Ensemble

"""

import numpy as np
from mlens.parallel.evaluation import fit_score
from mlens.utils.dummy import OLS, Data
from mlens.metrics import mape, make_scorer

X, y = Data('stack', False, False).get_data((10, 2), 3)


def test_fit_score():
    """[Parallel | Evaluation] Test fit-score function."""
    out = fit_score(case='test',
                    tr_list=[],
                    est_name='ols',
                    est=OLS(),
                    params=(0, {'offset': 2}),
                    x=X,
                    y=y,
                    idx=((0, 5), (5, 10)),
                    scorer=make_scorer(mape, greater_is_better=False),
                    error_score=None)

    assert out[0] == 'test'
    assert out[1] == 'ols'
    assert out[2] == 0

    np.testing.assert_almost_equal(out[3], -1.5499999999999992, 5)
    np.testing.assert_almost_equal(out[4], -2.0749999999999993, 5)
