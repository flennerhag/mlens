"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Formatting of instance lists.
"""

from __future__ import division, print_function

from .checks import assert_valid_estimator
from .exceptions import LayerSpecificationError
from collections import Counter


def _format_instances(instances):
    """Format a list of instances to a list of named estimator tuples."""
    named_instances = []
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
            if instance == val:
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

        # Check that the first element is a string with no spaces,
        # the latter an estimator
        is_str = isinstance(element[0], str)
        no_spa = ' ' not in element[0]
        is_est = (hasattr(element[1], 'get_params') and
                  hasattr(element[1], 'fit'))
        if not (is_str and is_est and no_spa):
            return False

        # Check that the last element is a valid estimator
        assert_valid_estimator(element[1])

    # Check that there are no duplicate names
    names = Counter([tup[0] for tup in instance_list])
    if max([val for val in names.values()]) > 1:
        return False

    # If instances passes above criterion, it's correctly specified
    return True


def _assert_format(instances):
    """Assert that a generic instances object is correctly formatted."""
    if isinstance(instances, dict):
        # Need to check every instance list across preprocessing cases
        for instance_list in instances.values():
            if not _check_format(instance_list):
                return False
        return True

    # For list, check the given list
    return _check_format(instances)


def check_instances(instances):
    """Helper to ensure all instances are named.

    Check if ``instances`` is formatted as expected, and if not convert
    formatting or throw traceback error if impossible to anticipate formatting.

    Parameters
    ----------
    instances : iterable
        instance iterable to test.

    Returns
    -------
    formatted : list or dict
        formatted ``instances`` object. Will be formatted as a dict if
        preprocessing cases are detected, otherwise as a list. The dict will
        contain lists identical to those in the single preprocessing case.
        Each list is of the form ``[('name', instance]`` and no names overlap.

    Raises
    ------
    LayerSpecificationError :
        Raises error if formatting fails, which is most likely due to wrong
        ordering of tuple entries, or wrong argument in the wrong position.

    See Also
    --------
        :class:`mlens.ensemble.base.Layer`
    """
    is_iterable = isinstance(instances, (list, tuple, dict))
    if instances is None or is_iterable and len(instances) == 0:
        # If no instances specified, return empty list
        return []
    elif not is_iterable:
        # Instance is the estimator, wrap in list and continue
        instances = [instances]

    if _assert_format(instances):
        # If format is ok, return as is
        return instances
    else:
        # reformat
        if isinstance(instances, dict):
            # We need to check the instance list of each case
            out = {}
            for case, case_list in instances.items():
                out['-'.join(case.lower().split())] = \
                    _format_instances(case_list)
            return out
        else:
            return _format_instances(instances)
