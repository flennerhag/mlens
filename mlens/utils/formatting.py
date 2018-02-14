"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Formatting of instance lists.
"""

from __future__ import division, print_function

from collections import Counter
from .checks import assert_valid_estimator, assert_correct_format
from .exceptions import LayerSpecificationError
from ..externals.sklearn.base import clone


def format_name(name, cls, global_names):
    """Utility for assigning a unique name"""
    # TODO: should pop names when instance is destroyed and check if name taken
    if not name:
        name = '%s-%i' % (cls, len(global_names))
        global_names.append(name)
    return name


def _format_instances(instances, namespace=None):
    """Format a list of instances to a list of named estimator tuples."""
    nested = isinstance(instances, dict)
    if nested:
        # Need to flatten, but record hierarchy
        instances_dict = instances
        vacuous = list()
        case_map = dict()
        instances = list()
        for case, instance_list in sorted(instances_dict.items()):
            case = '-'.join(case.lower().split())

            if not instance_list:
                vacuous.append((case, instance_list))
                continue

            for val in instance_list:
                instances.append(val)
                if isinstance(val, (list, tuple, set)):
                    val = val[1]  # Grab estimator
                case_map[val] = case

    named_instances = list()
    for val in instances:
        # Check that the instance appears correctly specified
        if not isinstance(val, (list, tuple, set)):
            # val is the instance
            instance = val
        else:
            # val is a list-like object. Assume instance is the last entry
            instance = val[-1]

        # Check if it appears to be an estimator
        assert_valid_estimator(instance)

        try:
            # Format instance names

            # We keep the instance as a list to change possible duplicate names
            # exploiting that lists are mutable, before switching to tuple
            if instance is val:
                tup = [instance.__class__.__name__.lower(), instance]
            else:
                tup = ['-'.join(val[0].split()).lower(), val[-1]]

            named_instances.append(tup)

        except Exception as e:
            msg = ("Could not format instance %s. Check that passed instance "
                   "iterables follow correct syntax:\n"
                   "- if multiple preprocessing cases, pass a dictionary with "
                   "instance lists as values and case name as key.\n"
                   "- else, pass list of (named) instances.\n"
                   "See documentation for further information.\n"
                   "Error details: %r")
            raise LayerSpecificationError(msg % (instance, e))

    # Check and correct duplicate names
    names = [tup[0] for tup in named_instances]
    if namespace:
        names.extend(namespace)
    duplicates = Counter(names)
    duplicates = {key: val for key, val in duplicates.items() if val > 1}

    out = list()  # final named_instances list
    name_count = {key: 1 for key in duplicates}
    for name, instance in named_instances:
        if name in duplicates:
            current_name_count = name_count[name]  # fix before update
            name_count[name] += 1
            name += '-%d' % current_name_count  # rename
        out.append((name, instance))
    out = sorted(out)

    if nested:
        # Rebuild hierarchy
        nested_out = dict()
        for name, instance in out:
            case = case_map[instance]
            if case not in nested_out:
                nested_out[case] = list()
            nested_out[case].append((name, instance))

        for k, v in vacuous:
            nested_out[k] = v

        out = nested_out

    return out


def _check_format(instance_list, namespace=None):
    """Quick check of an instance list to see if the format is correct."""
    if namespace is None:
        namespace = []
    # Assert list instance
    if not isinstance(instance_list, list):
        return False

    # If empty list, no preprocessing case
    if not instance_list:
        return True

    # Check if each element in instance_list is a named instance tuple
    for element in instance_list:

        # Check that element is a tuple
        if not isinstance(element, tuple) or len(element) != 2:
            return False

        # Check tuple
        is_str = isinstance(element[0], str)
        no_spa = ' ' not in element[0]
        no_dup = element[0] not in namespace

        is_est = (hasattr(element[1], 'get_params') and
                  hasattr(element[1], 'fit'))

        if not (is_str and is_est and no_dup and no_spa):
            return False

        # Check that the last element is a valid estimator
        assert_valid_estimator(element[1])

    # Check that there are no duplicate names
    names = Counter([tup[0] for tup in instance_list])
    if max([val for val in names.values()]) > 1:
        return False

    # If instances passes above criterion, it's correctly specified
    return True


def _assert_format(instances, namespace=None):
    """Assert that a generic instances object is correctly formatted."""
    if not namespace:
        namespace = list()

    if isinstance(instances, dict):
        # Need to check every instance list across preprocessing cases
        for instance_list in instances.values():
            if not _check_format(instance_list, namespace=namespace):
                return False
            namespace.extend(n for n, e in instance_list)
        return True

    # For list, check the given list
    return _check_format(instances, namespace=namespace)


def check_instances(estimators=None, preprocessing=None, namespace=None):
    """Ensure estimators and preprocessing pipelines are correctly formatted

    Utility for formating estimator iterable and preprocessing iterable into
    formats accepted by a the :class:`Layer`.

    Parameters
    ----------
    estimators : iterable, optional
        estimator instances.

    preprocessing : iterable, optional
        preprocessing pipeline instances.
    """
    if not namespace:
        namespace = list()
    assert_correct_format(estimators, preprocessing)
    preprocessing = _check_instances(preprocessing)
    if estimators is not None and not isinstance(estimators, (dict, list)):
        estimators = [estimators]

    if preprocessing:
        if isinstance(preprocessing, list):
            # Preprocessing but there's not case: force create
            preprocessing = {'pr': preprocessing}
            estimators = {'pr': estimators} if estimators else dict()
        preprocessing = [(n, l) for n, l in sorted(preprocessing.items())]
        namespace += [n for n, l in preprocessing]

    if estimators:
        estimators = _check_instances(estimators, namespace=namespace)
        estimators = _flatten(estimators)

    out_prep, out_est, cases = list(), list(), list()
    if preprocessing:
        for preprocess_name, tr in sorted(preprocessing):
            if tr:
                out_prep.append((preprocess_name,
                                 [(n, clone(t)) for n, t in tr]))
                cases.append(preprocess_name)
    if estimators:
        for preprocess_name, learner_name, est in estimators:
            pr_name = preprocess_name if preprocess_name in cases else None
            out_est.append((pr_name, learner_name, clone(est)))

    return out_prep, out_est


def _flatten(instances):
    """Flatten iterator"""
    # Flattened version
    if isinstance(instances, list):
        flattened = [(None, name, est) for name, est in sorted(instances)]
    else:
        # Compress dictionary and sort on case_est keys
        vps = [('%s__%s' % (case, est_name), est)
               for case, instance_list in instances.items()
               for est_name, est in instance_list]
        flattened = [(name.split('__')[0], name.split('__')[1], est)
                     for name, est in sorted(vps)]
    return flattened


def _check_instances(instances, namespace=None):
    """Helper to ensure all instances are named.

    Check if ``instances`` is formatted as expected, and if not convert
    formatting or throw traceback error if impossible to anticipate formatting.

    Parameters
    ----------
    instances : iterable
        instance iterable to test.

    namespace: list, optional
        list of assigned instance names.

    Returns
    -------
    formatted : list or dict
        formatted ``instances`` object. Will be formatted as a dict if
        preprocessing cases are detected, otherwise as a list. The dict will
        contain lists identical to those in the single preprocessing case.
        Each list is of the form ``[('name', instance)]`` and no names overlap.

    Raises
    ------
    LayerSpecificationError :
        Raises error if formatting fails, which is most likely due to wrong
        ordering of tuple entries, or wrong argument in the wrong position.

    See Also
    --------
        :class:`mlens.ensemble.base.Layer`
    """
    if not namespace:
        namespace = list()

    is_iterable = isinstance(instances, (list, tuple, dict))
    if not is_iterable:
        instances = [instances]
    if not instances or None in instances:
        return None

    if _assert_format(instances, namespace):
        out = instances
    else:
        out = _format_instances(
            instances, namespace=namespace)
    return out
