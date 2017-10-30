"""ML-ENSEMBLE

author: Sebastian Flennerhag
:copyirgh: 2017
:licence: MIT
"""
import numpy as np

from mlens.externals.sklearn.base import clone
from mlens.utils.checks import check_ensemble_build, assert_valid_estimator, \
    assert_correct_format, check_initialized

from mlens.utils.dummy import OLS, Scale
from mlens.testing import EstimatorContainer
from mlens.utils.exceptions import LayerSpecificationError, \
    LayerSpecificationWarning, NotFittedError, ParallelProcessingError, \
    ParallelProcessingWarning

lg = EstimatorContainer()

LAYER = lg.get_layer('stack', False, True)
LAYER_CONTAINER = lg.get_sequential('stack', False, True)


class Tmp(object):

    """Temporary class for mimicking ParallelProcessing status."""

    def __init__(self, lyr, __initialized__, __fitted__):
        self.__initialized__ = __initialized__
        self.__fitted__ = __fitted__
        self.layers = lyr


class Lyr(object):

    """Temporary layer class for storing raise on exception."""

    def __init__(self, raise_on_exception):
        self.raise_on_exception = raise_on_exception


def test_check_ensemble_build_passes():
    """[Utils] check_ensemble_build : passes on default dummy LC."""
    FLAG = check_ensemble_build(LAYER_CONTAINER)
    assert FLAG is True


def test_check_ensemble_build_no_lc():
    """[Utils] check_ensemble_build : raises error on no LC."""
    lc = clone(LAYER_CONTAINER)
    del lc.stack
    np.testing.assert_raises(AttributeError, check_ensemble_build, lc)


def test_check_ensemble_build_lc_none():
    """[Utils] check_ensemble_build : raises error on LC None."""
    lc = clone(LAYER_CONTAINER)
    lc.raise_on_exception = True
    lc.stack = None

    np.testing.assert_raises(LayerSpecificationError, check_ensemble_build, lc)


def test_check_ensemble_build_lc_none_no_raise_():
    """[Utils] check_ensemble_build : raises warning on LC None + no raise_."""
    lc = clone(LAYER_CONTAINER)
    lc.stack = None
    lc.raise_on_exception = False

    FLAG = np.testing.assert_warns(LayerSpecificationWarning,
                                   check_ensemble_build, lc)
    assert FLAG is False


def test_assert_valid_estimator():
    """[Utils] assert_valid_estimator: check passes valid estimator."""
    assert_valid_estimator(OLS())


def test_assert_valid_estimator_fails_class():
    """[Utils] assert_valid_estimator: check fails uninstantiated estimator."""
    np.testing.assert_raises(TypeError, assert_valid_estimator, OLS)


def test_assert_valid_estimator_fails_type():
    """[Utils] assert_valid_estimator: check fails non-estimator."""
    np.testing.assert_raises(TypeError, assert_valid_estimator, 1)


def test_assert_correct_layer_format_1():
    """[Utils] assert_correct_format: prep - none, est - list."""
    assert_correct_format([OLS()], [])


def test_assert_correct_layer_format_2():
    """[Utils] assert_correct_format: prep - list, est - list."""
    assert_correct_format([OLS()], [Scale()])


def test_assert_correct_layer_format_3():
    """[Utils] assert_correct_format: prep - dict, est - dict."""
    assert_correct_format({'a': [OLS()]}, {'a': [Scale()]})


def test_assert_correct_layer_format_4():
    """[Utils] assert_correct_format: prep - inst, est - inst."""
    assert_correct_format(OLS(), Scale())


def test_assert_correct_layer_format_fails_dict_list():
    """[Utils] assert_correct_format: prep - list, est - dict."""
    np.testing.assert_raises(LayerSpecificationError,
                             assert_correct_format,
                             {'a': [OLS()]}, [])


def test_assert_correct_layer_format_fails_dict_none():
    """[Utils] assert_correct_format: prep - none, est - dict."""
    np.testing.assert_raises(LayerSpecificationError,
                             assert_correct_format,
                             {'a': [OLS()]}, None)


def test_assert_correct_layer_format_tuple():
    """[Utils] assert_correct_format: prep - dict, est - list."""
    np.testing.assert_raises(LayerSpecificationError,
                             assert_correct_format,
                             OLS(), {'a': [Scale()]})


def test_assert_correct_layer_format_dict_keys():
    """[Utils] assert_correct_format: assert raises on no key overlap."""
    np.testing.assert_raises(LayerSpecificationError,
                             assert_correct_format,
                             {'a': [OLS()]}, {'b': [Scale()]})


def test_check_initialized():
    """[Utils] check_initialized: passes initialized."""
    check_initialized(Tmp(Lyr(True), 1, 0))


def test_check_initialized_fails():
    """[Utils] check_initialized: fails not initialized."""
    np.testing.assert_raises(ParallelProcessingError,
                             check_initialized, Tmp(Lyr(True), 0, 0))


def test_check_initialized_fails_fitted():
    """[Utils] check_initialized: fails initialized and fitted."""
    np.testing.assert_raises(ParallelProcessingError,
                             check_initialized, Tmp(Lyr(True), 1, 1))


def test_check_initialized_warns_fitted():
    """[Utils] check_initialized: warns initialized and fitted if not raise."""
    np.testing.assert_warns(ParallelProcessingWarning,
                            check_initialized, Tmp(Lyr(False), 1, 1))
