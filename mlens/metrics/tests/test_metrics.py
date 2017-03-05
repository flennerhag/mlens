"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 03/03/2017
"""

from __future__ import division, print_function

import mlens.metrics as metrics
import numpy as np

SEED = 100

np.random.seed(SEED)

y = np.random.random(100)
p = np.random.random(100)
z = np.random.random(100)


def test_score_matrix():
    out = metrics.score_matrix(np.vstack([p, z]).T, y, metrics.metrics.rmse_scoring, column_names=None, prefix=None)

    assert set(out.keys()) == ['pred_0', 'pred_1']
    assert out['pred_0'] == '0.421034921019'
    assert out['pred_1'] == '0.429606386793'


def test_score_matrix_prefix():
    out = metrics.score_matrix(np.vstack([p, z]).T, y, metrics.metrics.rmse_scoring, prefix='test')

    assert isinstance(out, dict)
    assert set(out.keys()) == set(['test-pred_0', 'test-pred_1'])

test_score_matrix()
test_score_matrix_prefix()