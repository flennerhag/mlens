"""ML-ENSEMBLE

author: Sebastian Flennerhag
licence: MIT
Support functions used throughout mlens
"""

from __future__ import division, print_function

from ..utils.checks import SliceError, LayerSpecificationError
from numpy import ix_
from collections import Counter
from sklearn.pipeline import Pipeline, _name_estimators


###############################################################################
def safe_slice(X, row_slice=None, column_slice=None, layer_name=None):
    """Safe slice of X irrespective of whether X is DataFrame or ndarray."""
    if (row_slice is None) and (column_slice is None):
        # No slice specified return original
        return X
    else:
        # Slice X
        try:
            # Wrap slicing in a try-except block to wrap traceback
            if hasattr(X, "iloc"):
                # Check if pandas DataFrame
                if row_slice is None:
                    return X.iloc[:, column_slice]
                elif column_slice is None:
                    return X.iloc[row_slice, :]
                else:
                    return X.iloc[row_slice, column_slice]

            else:
                # Assume numpy array, or similarly indexable
                if row_slice is None:
                    return X[:, column_slice]
                elif column_slice is None:
                    return X[row_slice]
                else:
                    return X[ix_(row_slice, column_slice)]

        except Exception as e:
            # Throw error along with some information about X
            if layer_name is not None:
                # Print passed layer information
                raise SliceError(
                    '[%s] Slicing array failed. Aborting. '
                    'Details:\n%r\nX: %s\n%r' % (layer_name, e, type(X), X))
            else:
                raise SliceError('Slicing array failed. Aborting. '
                                 'Details:\n%r\nX: %s\n%r' % (e, type(X), X))


###############################################################################
def check_fit_overlap(full_fit_est, fold_fit_est, layer):
    """Helper function to check that fitted estimators overlap."""
    if not all([est in full_fit_est for est in fold_fit_est]):
        raise ValueError('[%s] Not all estimators successfully fitted on the '
                         'full data set were fitted during fold predictions. '
                         'Aborting.'
                         '\n[%s] Fitted estimators on full data: %r'
                         '\n[%s] Fitted estimators on folds:'
                         '%r' % (layer, layer, full_fit_est, layer,
                                 fold_fit_est))

    if not all([est in fold_fit_est for est in full_fit_est]):
        raise ValueError('[%s] Not all estimators successfully fitted on the '
                         'fold data were successfully fitted on the full data.'
                         ' Aborting.'
                         '\n[%s] Fitted estimators on full data: %r'
                         '\n[%s] Fitted estimators on folds:'
                         '%r' % (layer, layer, full_fit_est, layer,
                                 fold_fit_est))


def name_columns(estimator_cases):
    """Utility func for naming a mapping of estimators on different cases."""
    return [case + '-' + est_name if case not in [None, ''] else est_name
            for case, estimators in estimator_cases.items()
            for est_name, _ in estimators]


def name_estimators(estimators, prefix='', suffix=''):
    """Function for creating dict with named estimators for get_params."""
    if len(estimators) == 0:
        return {}
    else:
        if isinstance(estimators[0], tuple):
            # if first item is a tuple, assumed list of named tuple was passed
            named_estimators = {prefix + est_name + suffix: est for
                                est_name, est in estimators}
        else:
            # assume a list of unnamed estimators was passed
            named_estimators = {prefix + est_name + suffix: est for
                                est_name, est in _name_estimators(estimators)}
        return named_estimators


def name_layer(layer, layer_prefix=''):
    """Function for naming layer for parameter setting."""
    if isinstance(layer, list):
        # if a list is passed, assume it is a list of base estimators
        return name_estimators(layer)
    else:
        # Assume preprocessing cases are specified
        named_layer = {}
        for p_name, pipe in layer.items():
            if isinstance(pipe, Pipeline):
                # If pipeline is passed, get the list of steps
                pipe = pipe.steps

            # Add prefix to estimators to uniquely define each layer and
            # preprocessing case
            if len(p_name) > 0:
                if len(layer_prefix) > 0:
                    prefix = layer_prefix + '-' + p_name + '-'
                else:
                    prefix = p_name + '-'
            else:
                prefix = layer_prefix

            for phase in pipe:
                named_layer.update(name_estimators(phase, prefix))

    return named_layer


def _format_instances(instances):
    """Format a list of instances to a list of named estimator tuples."""
    named_instances = []
    for val in instances:
        # Check that the instance appears correctly specified
        if not isinstance(val, (list, tuple, set)):
            # val is the instance
            instance = val
            has_get_params = hasattr(instance, 'get_params')
            has_fit = hasattr(instance, 'fit')
        else:
            # val is a list-like object. Assume instance is the last entry
            instance = val[-1]
            has_get_params = hasattr(instance, 'get_params')
            has_fit = hasattr(instance, 'fit')

        if not has_get_params:
            raise TypeError('[%s] does not appear to be a valid'
                            ' estimator. as it does not implement a '
                            '`get_params` method. Type: '
                            '%s' % (instance, type(instance)))

        if not has_fit:
            raise TypeError('[%s] does not appear to be a valid'
                            ' estimator. as it does not implement a '
                            '`fit` method. Type: '
                            '%s' % (instance, type(instance)))
        try:
            # Name instance

            # We keep the tuple as a list to change possible duplicate names
            # before switching to tuple
            if instance == val:
                tup = _name_estimators([instance])[0]
            else:
                tup = (val[0].lower(), val[-1])

            named_instances.append(tup)

        except Exception as e:
            msg = ('Layer instantiation failed due to incorrectly '
                   'specification. Check that estimators and preprocessing '
                   'follows correct syntax:\n'
                   '- if preprocessing cases, pass dictionaries of instance '
                   'lists.\n'
                   '- else, pass instance list(s).\n'
                   'See documentation for further information.\n'
                   'Instance failure:\n%r\nError details: %r')
            raise LayerSpecificationError(msg % (instance, e))

    # Check and correct duplicate names
    duplicates = Counter([tup[0] for tup in named_instances])
    duplicates = {key: val for key, val in duplicates.items() if
                  val > 1}

    out = []  # final named_instances list

    name_count = {key: 1 for key in duplicates}
    for name, instance in named_instances:
        if name in duplicates:
            current_name_count = name_count[name]  # fix before update
            name_count[name] += 1
            name += '-%d' % current_name_count  # rename
        out.append((name, instance))

    return out


def _check_format(instance_list):
    """Quick check of an instance list to see if the format is correct."""
    # Assert list instance
    if not isinstance(instance_list, list):
        return False

    # If empty list, no preprocessing case
    if len(instance_list) == 0:
        return True

    # Check if each element in instance_list is a named instance tuple
    for element in instance_list:

        # Check that element is a tuple
        if not isinstance(element, tuple) or len(element) != 2:
            return False

        # Check that the first element is a string, the latter an estimator
        is_str = isinstance(element[0], str)
        is_est = (hasattr(element[1], 'get_params') and
                  hasattr(element[1], 'fit'))
        if not (is_str and is_est):
            return False

    # Check that there are no duplicate names
    names = Counter([tup[0] for tup in instance_list])
    if max([val for val in names.values()]) > 1:
        return False

    # If 'instances' passes above criterion, it's correctly specified
    return True


def _assert_format(instances):
    """Assert that a generic instances object is correctly formatted."""
    if isinstance(instances, dict):
        # Need to check every instance list across preprocessing cases
        for instance_list in instances.values():
            if not _check_format(instance_list):
                return False
        return True
    else:
        # Check the given list
        return _check_format(instances)


def check_instances(instances):
    """Helper to ensure all instances are named.

    Check if 'instances' is formatted as expected, and if not convert
    formatting or throw traceback error if impossible to anticipate formatting.

    Parameters
    ----------
    instances: iterable
        instance iterable to test.

    Returns
    -------
    formatted: list, dict
        formatted 'instances' object. Will be formatted as a dict if
        preprocessing cases are detected, otherwise as a list. The dict will
        contain lists identical to those in the single preprocessing case.
        Each list is of the form [('name', instance] and no names overlap.
    """
    if (instances is None) or (len(instances) is 0):
        # If no instances specified, return empty list
        return []
    elif _assert_format(instances):
        # If format is ok, return as is
        return instances
    else:
        # reformat
        if isinstance(instances, dict):
            # We need to check the instance list of each case
            out = {}
            for case, case_list in instances.items():
                out[case] = _format_instances(case_list)
            return out
        else:
            return _format_instances(instances)
