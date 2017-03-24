"""ML-ENSEMBLE

:author: Sebastian Flennerhag

Nosetests for :class:`mlens.base`
"""

import numpy as np

from mlens.base import IdTrain, FoldIndex, BlendIndex

SEED = 100
np.random.seed(SEED)

X = np.array(range(25)).reshape(5, 5)


###############################################################################
def test_id_train():
    """[Base] Test IdTrain class for checking training and test matrices."""
    id_train = IdTrain(size=4)
    id_train.fit(X)

    assert id_train.is_train(X)
    assert not id_train.is_train(np.random.permutation(X))
    assert not id_train.is_train(X[:3, :])


###############################################################################
def test_full_index_is_fitted():
    """[Base] FoldIndex: check fit methods."""
    idx = FullIndex(4)
    assert not hasattr(idx, 'n_samples')
    idx.fit(X)
    assert hasattr(idx, 'n_samples')

    idx = FullIndex(4)
    assert not hasattr(idx, 'n_samples')
    for _ in idx.generate(X): pass
    assert hasattr(idx, 'n_samples')

    idx = FullIndex(4, X)
    assert hasattr(idx, 'n_samples')


def test_full_tuple_shape():
    """[Base] FoldIndex: test the tuple shape on generation."""
    tups = [(tri, tei) for tri, tei in FullIndex(5, X).generate()]

    assert tups == [(((1, 5),)       , (0, 1)),
                    (((0, 1), (2, 5)), (1, 2)),
                    (((0, 2), (3, 5)), (2, 3)),
                    (((0, 3), (4, 5)), (3, 4)),
                    (((0, 4), (5, 5)), (4, 5))
                    ]


def test_full_array_shape():
    """[Base] FoldIndex: test the array shape on generation."""
    tr = [np.array([2, 3, 4]),  np.array([0, 1, 4]), np.array([0, 1, 2, 3])]
    te = [np.array([0, 1]), np.array([2, 3]), np.array([4])]

    for i, (tri, tei) in enumerate(FullIndex(3, X).generate(as_array=True)):

        np.testing.assert_array_equal(tri, tr[i])
        np.testing.assert_array_equal(tei, te[i])


def test_full_raises_on_oversampling():
    """[Base] FoldIndex: check raises error."""
    with np.testing.assert_raises(ValueError):
        FullIndex(100, X)


def test_full_raises_on_fold_1():
    """[Base] FoldIndex: check raises error on folds=1."""
    with np.testing.assert_raises(ValueError):
        FullIndex(1, X)


def test_full_warns_on_fold_1():
    """[Base] FoldIndex: check warns on folds=1 if not raise_on_exception."""
    with np.testing.assert_warns(UserWarning):
        FullIndex(1, X, raise_on_exception=False)


def test_full_raises_on_float():
    """[Base] FoldIndex: check raises error on float."""
    with np.testing.assert_raises(ValueError):
        FullIndex(0.5, X)


def test_full_raises_on_empty():
    """[Base] FoldIndex: check raises error on singular array."""
    with np.testing.assert_raises(ValueError):
        FullIndex(2, np.empty(1))


###############################################################################
def test_blend_index_is_fitted():
    """[Base] BlendIndex: check fit methods."""
    attrs = ['n_samples', 'n_train', 'n_test']

    idx = BlendIndex(2, 3)
    for attr in attrs: assert not hasattr(idx, attr)
    idx.fit(X)
    for attr in attrs: assert hasattr(idx, attr)

    idx = BlendIndex(2, 3)
    for attr in attrs: assert not hasattr(idx, attr)
    for _ in idx.generate(X): pass
    for attr in attrs: assert hasattr(idx, attr)

    idx = BlendIndex(2, 3, X)
    for attr in attrs: assert hasattr(idx, attr)

def test_blend_tuple_shape():
    """[Base] BlendIndex: test the tuple shape on generation."""
    tup = [(tri, tei) for tri, tei in BlendIndex(0.45, 0.55).generate(X)][0]

    assert tup == ((0, 2), (2, 4))


def test_blend_array_shape():
    """[Base] BlendIndex: test the array shape on generation."""
    tr = np.array([0, 1])
    te = np.array([2, 3])

    for tri, tei in BlendIndex(0.45, 0.55, X).generate(as_array=True):

        np.testing.assert_array_equal(tri, tr)
        np.testing.assert_array_equal(tei, te)


def test_blend_raises_on_oversampling():
    """[Base] BlendIndex: check raises error on float sums > 1.0."""
    for tup in [(1.2, None), (0.6, 0.6)]:
        with np.testing.assert_raises(ValueError):
            BlendIndex(*tup, X=X)


def test_blend_raises_on_empty():
    """[Base] BlendIndex: check raises error on singular array."""
    with np.testing.assert_raises(ValueError):
        BlendIndex(2, 2, np.empty(1))


def test_blend_raises_empty_test():
    """[Base] BlendIndex: check raises error on empty test set."""
    with np.testing.assert_raises(ValueError):
        BlendIndex(0, 10, X)


def test_blend_raises_empty_train():
    """[Base] BlendIndex: check raises error on empty train set."""
    with np.testing.assert_raises(ValueError):
        BlendIndex(10, 0, X)
