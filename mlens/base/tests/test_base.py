"""ML-ENSEMBLE

author: Sebastian Flennerhag
"""

import numpy as np
from pandas import DataFrame

from mlens.utils.exceptions import SliceError
from mlens.base import (clone_base_estimators, clone_preprocess_cases,
                        name_estimators, name_layer, check_instances,
                        check_fit_overlap, name_columns, safe_slice, IdTrain)

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

SEED = 100
np.random.seed(SEED)

X = np.array(range(25)).reshape(5, 5)
Xdf = DataFrame(X)
Z = np.array(range(15)).reshape(3, 5)

id_train = IdTrain(size=4)

# Some meta estimator
meta = SVR()

# Some pipeline to be passed to the `.add` method
layer = {
    'sc':
        ([StandardScaler()],
         [('ls', Lasso()), ('kn', KNeighborsRegressor())]),
    'mm':
        ([MinMaxScaler()], [SVR()]),
    'np':
        ([], [('rf', RandomForestRegressor(random_state=100))])
        }


def test_naming():
    """[Base] Test correct naming of estimators."""
    named_meta = name_estimators([meta], 'meta-')
    named_base = name_layer(layer)

    assert isinstance(named_meta, dict)
    assert isinstance(named_meta['meta-svr'], SVR)
    assert isinstance(named_base, dict)
    assert len(named_base) == 6


def test_check_instances():
    """[Base] Test that unnamed estimator lists are named."""
    preprocess = [(case, check_instances(p[0])) for case, p in
                  layer.items()]

    base_estimators = [(case, check_instances(p[1])) for case, p in
                       layer.items()]

    assert isinstance(base_estimators, list)
    assert isinstance(preprocess, list)
    assert len(base_estimators) == 3
    assert len(preprocess) == 3
    assert isinstance(base_estimators[0], tuple)
    assert isinstance(preprocess[0], tuple)


def test_clone():
    """[Base] Preprocess and estimator pipes clone and return handling."""
    preprocess = {case: check_instances(p[0]) for case, p in
                  layer.items()}
    base_estimators = {case: check_instances(p[1]) for case, p in
                       layer.items()}

    base_ = clone_base_estimators(base_estimators)
    base_list = clone_base_estimators(base_estimators, as_dict=False)

    preprocess_ = clone_preprocess_cases(preprocess)
    preprocess_list = clone_preprocess_cases(None)

    base_columns_ = name_columns(base_)

    assert preprocess_list is None
    assert isinstance(preprocess_, list)
    assert isinstance(preprocess_[0], tuple)
    assert isinstance(preprocess_[0][1], list)
    assert isinstance(base_, dict)
    assert isinstance(base_['mm'], list)
    assert isinstance(base_['mm'][0], tuple)
    assert isinstance(base_columns_, list)
    assert len(base_columns_) == 4

    key_list = [tup[0] for tup in base_list]
    for key in base_:
        assert key in key_list


def test_check_estimators():
    """[Base] Test that fitted estimator overlap is correctly checked."""
    fold_fit_incomplete = ['a']
    fold_fit_complete = ['a', 'b']

    full_fit_incomplete = ['a']
    full_fit_complete = ['a', 'b']

    check_fit_overlap(fold_fit_complete, full_fit_complete, 'layer-1')

    try:
        check_fit_overlap(fold_fit_incomplete, full_fit_complete, 'layer-1')
    except ValueError as e:
        assert issubclass(type(e), ValueError)
        assert str(e) == \
               ("[layer-1] Not all estimators successfully fitted on "
                "the full data set were fitted during fold predictions. "
                "Aborting.\n[layer-1] Fitted estimators on full data: ['a']\n"
                "[layer-1] Fitted estimators on folds:['a', 'b']")

    try:
        check_fit_overlap(fold_fit_complete, full_fit_incomplete, 'layer-1')
    except ValueError as e:
        assert issubclass(type(e), ValueError)
        assert str(e) == \
               ("[layer-1] Not all estimators successfully fitted on the fold "
                "data were successfully fitted on the full data. Aborting.\n"
                "[layer-1] Fitted estimators on full data: ['a', 'b']\n"
                "[layer-1] Fitted estimators on folds:['a']")


def test_column_naming():
    """[Base] Assert correct columns naming."""
    cols = name_columns({'case-1': [('est-1', SVR()), ('est-2', Lasso())]})

    for i, key in enumerate(cols):
        assert key == 'case-1-est-%i' % (i + 1)


def test_safe_slice():
    """[Base] Test safe slicing correctly retrieves subsets from X and Xdf."""
    # Test sets
    row_slice = np.array([[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]])
    row_slice_df = DataFrame(row_slice, index=[2, 3])

    column_slice = np.array([[2,  3], [7,  8], [12, 13], [17, 18], [22, 23]])
    column_slice_df = DataFrame(column_slice, columns=[2, 3])

    full_slice = np.array([[12, 13], [17, 18]])
    full_slice_df = DataFrame(full_slice, index=[2, 3], columns=[2, 3])

    # Sliced samples
    row_sample = safe_slice(X, row_slice=[2, 3])
    row_sample_df = safe_slice(Xdf, row_slice=[2, 3])

    column_sample = safe_slice(X, column_slice=[2, 3])
    column_sample_df = safe_slice(Xdf, column_slice=[2, 3])

    full_sample = safe_slice(X, [2, 3], [2, 3])
    full_sample_df = safe_slice(Xdf, [2, 3], [2, 3])

    # Assert correct numpy array slices
    np.testing.assert_array_equal(row_slice, row_sample)
    np.testing.assert_array_equal(column_slice, column_sample)
    np.testing.assert_array_equal(full_slice, full_sample)

    # Assert correct DataFrame slices
    for df in [column_sample_df, row_sample_df, full_sample_df]:
        assert isinstance(df, DataFrame)

    assert row_slice_df.equals(row_sample_df)
    assert column_slice_df.equals(column_sample_df)
    assert full_slice_df.equals(full_sample_df)


def test_no_index():
    """[Base] Check that safe_slice with no passed slice return input."""
    foo = {}

    # No slicing should return object
    out = safe_slice(foo)
    assert id(out) == id(foo)


def test_wrong_type_slice():
    """[Base] Check that safe_slice raise error on wrong type of input."""
    wrong_type = {'wrong_type': None}

    try:
        safe_slice(wrong_type, [0, 1])
        raise AssertionError('Error: Slicing a dictionary passed!')
    except Exception as e:
        assert issubclass(type(e), SliceError)
        assert str(e) == \
               ('Slicing array failed. Aborting. Details:\nTypeError'
                '("unhashable type: \'list\'",)\nX: <class \'dict\'>\n'
                '{\'wrong_type\': None}')


def test_index_error_ndarray():
    """[Base] Check that safe_slice raise error over-indexing ndarray."""
    try:
        safe_slice(X, range(100), layer_name='layer-1')
        raise AssertionError('Error: Slicing out-of-bounds ndarray passed!')
    except Exception as e:
        assert issubclass(type(e), SliceError)
        assert str(e) == \
               ("[layer-1] Slicing array failed. Aborting. Details:\n"
                "IndexError('index 5 is out of bounds for axis 0 with "
                "size 5',)\nX: <class 'numpy.ndarray'>\n"
                "array([[ 0,  1,  2,  3,  4],\n"
                "       [ 5,  6,  7,  8,  9],\n"
                "       [10, 11, 12, 13, 14],\n"
                "       [15, 16, 17, 18, 19],\n"
                "       [20, 21, 22, 23, 24]])")


def test_safe_slice_error_raise():
    """[Base] Check that safe_slice raise error over-indexing DataFrame."""
    try:
        safe_slice(Xdf, range(100), layer_name='layer-1')
        raise AssertionError('Error: Slicing out-of-bounds DataFrame passed!')
    except Exception as e:
        assert issubclass(type(e), SliceError)
        assert str(e) == \
               ("[layer-1] Slicing array failed. Aborting. Details:\n"
                "IndexError('positional indexers are out-of-bounds',)\n"
                "X: <class 'pandas.core.frame.DataFrame'>\n"
                "    0   1   2   3   4\n"
                "0   0   1   2   3   4\n"
                "1   5   6   7   8   9\n"
                "2  10  11  12  13  14\n"
                "3  15  16  17  18  19\n"
                "4  20  21  22  23  24")


def test_id_train():
    """[Base] Test IdTrain class for checking training and test matrices."""
    id_train.fit(X)

    assert id_train.is_train(X)
    assert not id_train.is_train(np.random.permutation(X))
    assert not id_train.is_train(Z)

    id_train.fit(Xdf)

    assert id_train.is_train(Xdf)
    assert not id_train.is_train(DataFrame(np.random.permutation(X)))
    assert not id_train.is_train(DataFrame(Z))
