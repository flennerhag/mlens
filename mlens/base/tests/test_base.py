"""ML-ENSEMBLE

:author: Sebastian Flennerhag

Nosetests for :class:`mlens.base`
"""

import numpy as np

from mlens.base import IdTrain, FoldIndex, BlendIndex, SubsetIndex, FullIndex
from mlens.base.indexer import _partition, _prune_train

X = np.arange(25).reshape(5, 5)


###############################################################################
def test_id_train():
    """[Base] Test IdTrain class for checking training and test matrices."""
    id_train = IdTrain(size=5)
    id_train.fit(X)

    assert id_train.is_train(X)
    assert not id_train.is_train(np.random.permutation(X))
    assert not id_train.is_train(X[:3, :])


###############################################################################
def test_full_index():
    """[Base] FullIndex: check generates None."""
    for out in FullIndex(X=X).generate():
        assert out == (None, None)


###############################################################################
def test_full_index_is_fitted():
    """[Base] FoldIndex: check fit methods."""
    idx = FoldIndex(4)
    assert not hasattr(idx, 'n_samples')
    idx.fit(X)
    assert hasattr(idx, 'n_samples')

    idx = FoldIndex(4)
    assert not hasattr(idx, 'n_samples')
    for _ in idx.generate(X): pass
    assert hasattr(idx, 'n_samples')

    idx = FoldIndex(4, X)
    assert hasattr(idx, 'n_samples')


def test_full_tuple_shape():
    """[Base] FoldIndex: test the tuple shape on generation."""
    tups = [(tri, tei) for tri, tei in FoldIndex(5, X=X).generate()]

    assert tups == [(((1, 5),)       , (0, 1)),
                    (((0, 1), (2, 5)), (1, 2)),
                    (((0, 2), (3, 5)), (2, 3)),
                    (((0, 3), (4, 5)), (3, 4)),
                    (((0, 4),)       , (4, 5))
                    ]


def test_full_array_shape():
    """[Base] FoldIndex: test the array shape on generation."""
    tr = [np.array([2, 3, 4]),  np.array([0, 1, 4]), np.array([0, 1, 2, 3])]
    te = [np.array([0, 1]), np.array([2, 3]), np.array([4])]

    for i, (tri, tei) in enumerate(FoldIndex(3, X).generate(as_array=True)):

        np.testing.assert_array_equal(tri, tr[i])
        np.testing.assert_array_equal(tei, te[i])


def test_full_raises_on_oversampling():
    """[Base] FoldIndex: check raises error."""
    with np.testing.assert_raises(ValueError):
        FoldIndex(100, X)


def test_full_raises_on_fold_1():
    """[Base] FoldIndex: check raises error on folds=1."""
    with np.testing.assert_raises(ValueError):
        FoldIndex(1, X)


def test_full_warns_on_fold_1():
    """[Base] FoldIndex: check warns on folds=1 if not raise_on_exception."""
    with np.testing.assert_warns(UserWarning):
        FoldIndex(1, X, raise_on_exception=False)


def test_full_raises_on_float():
    """[Base] FoldIndex: check raises error on float."""
    with np.testing.assert_raises(ValueError):
        FoldIndex(0.5, X)


def test_full_raises_on_empty():
    """[Base] FoldIndex: check raises error on singular array."""
    with np.testing.assert_raises(ValueError):
        FoldIndex(2, np.empty(1))


###############################################################################
def test_blend_index_is_fitted():
    """[Base] BlendIndex: check fit methods."""
    attrs = ['n_samples', 'n_test_samples', 'n_train', 'n_test']

    idx = BlendIndex(2, 3)
    for attr in attrs: assert not hasattr(idx, attr)
    idx.fit(X)
    for attr in attrs: assert hasattr(idx, attr)

    idx = BlendIndex(2, 3)
    for attr in attrs: assert not hasattr(idx, attr)
    for _ in idx.generate(X): pass
    for attr in attrs: assert hasattr(idx, attr)

    idx = BlendIndex(2, 3, X=X)
    for attr in attrs: assert hasattr(idx, attr)


def test_blend_tuple_shape():
    """[Base] BlendIndex: test the tuple shape on generation."""
    tup = [(tri, tei) for tri, tei in BlendIndex(0.4, 0.5).generate(X)]

    assert tup == [((0, 2), (2, 4))]


def test_blend_array_shape():
    """[Base] BlendIndex: test the array shape on generation."""
    tr = np.array([0, 1])
    te = np.array([2, 3])

    for tri, tei in BlendIndex(0.45, 0.55, X=X).generate(as_array=True):

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
        BlendIndex(2, 2, X=np.empty(1))


def test_blend_raises_empty_test():
    """[Base] BlendIndex: check raises error on empty test set."""
    with np.testing.assert_raises(ValueError):
        BlendIndex(0, 10, X=X)


def test_blend_raises_empty_train():
    """[Base] BlendIndex: check raises error on empty train set."""
    with np.testing.assert_raises(ValueError):
        BlendIndex(10, 0, X=X)


###############################################################################
def test_subset_index_is_fitted():
    """[Base] BlendIndex: check fit methods."""
    attrs = ['n_samples', 'n_test_samples']

    idx = SubsetIndex()
    for attr in attrs: assert not hasattr(idx, attr)
    idx.fit(X)
    for attr in attrs: assert hasattr(idx, attr)

    idx = SubsetIndex()
    for attr in attrs: assert not hasattr(idx, attr)
    for _ in idx.generate(X): pass
    for attr in attrs: assert hasattr(idx, attr)

    idx = SubsetIndex(X=X)
    for attr in attrs: assert hasattr(idx, attr)


def test_subset_partition_array():
    """[Base] Subset: test partition indexing on arrays."""
    parts = []
    for part in SubsetIndex(X=X).partition(as_array=True):
        parts.append(part)

    np.testing.assert_array_equal(parts[0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(parts[1], np.array([3, 4]))


def test_subset_partition():
    """[Base] Subset: test partition indexing on tuples."""
    parts = []
    for part in SubsetIndex(X=X).partition():
        parts.append(part)

    assert parts == [(0, 3), (3, 5)]


def test_subset_tuple_shape():
    """[Base] BlendIndex: test the tuple shape on generation."""
    tup = [(tri, tei) for tri, tei in SubsetIndex(2, 2).generate(X)]

    assert tup == [(((2, 3),), [(0, 2), (3, 4)]),
                   (((0, 2),), [(2, 3), (4, 5)]),
                   (((4, 5),), [(0, 2), (3, 4)]),
                   (((3, 4),), [(2, 3), (4, 5)])]


def test_subset_array_shape():
    """[Base] SubsetIndex: test the array shape on generation."""

    t = []
    e = []
    for tri, tei in SubsetIndex(2, 2, X=X).generate(as_array=True):
        t.append(tri.tolist())
        e.append(tei.tolist())

    assert t == [[2], [0, 1], [4], [3]]
    assert e == [[0, 1, 3], [2, 4], [0, 1, 3], [2, 4]]


def test_subset_warns_on_wo_raise_():
    """[Base] SubsetIndex: check raises on n_part = 1, n_splits = 1."""
    with np.testing.assert_warns(UserWarning):
        SubsetIndex(1, 1, raise_on_exception=False, X=X)


def test_subset_raises_on_w_raise_():
    """[Base] SubsetIndex: check raises on n_part = 1, n_splits = 1."""
    with np.testing.assert_raises(ValueError):
        SubsetIndex(1, 1, X=X)


def test_subset_raises_on_float():
    """[Base] SubsetIndex: check raises error on floats for n_part, n_split."""
    with np.testing.assert_raises(ValueError):
        SubsetIndex(0.5, 2, X=X)

    with np.testing.assert_raises(ValueError):
        SubsetIndex(2, 0.5, X=X)


def test_subset_raises_on_partitions_and_one_split():
    """[Base] SubsetIndex: check raises error on single split of partitions."""
    with np.testing.assert_raises(ValueError):
        SubsetIndex(2, 1, X=X)


def test_subset_raises_on_no_split_part():
    """[Base] SubsetIndex: check raises error n_part * n_split > n_samples."""
    with np.testing.assert_raises(ValueError):
        SubsetIndex(3, 3, X=X)


def test_subset_raises_no_partition():
    """[Base] SubsetIndex: check raises error on 0 partitions."""
    with np.testing.assert_raises(ValueError):
        SubsetIndex(0, X=X)


def test_subset_raises_empty():
    """[Base] SubsetIndex: check raises error on empty train set."""
    with np.testing.assert_raises(ValueError):
        SubsetIndex(2, 2, X=np.empty(1))


###############################################################################
def test_prune_train():
    """[Base] indexers: test _prune_train."""
    assert _prune_train(0, 0, 4, 7) == ((4, 7),)
    assert _prune_train(4, 7, 9, 9) == ((4, 7),)
    assert _prune_train(4, 7, 10, 12) == ((4, 7), (10, 12))


def test_partition():
    """[Base] indexers: test _partition."""
    np.testing.assert_array_equal(np.array([4, 3, 3]), _partition(10, 3))

