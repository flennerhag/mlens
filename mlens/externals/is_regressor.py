#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""From sklearn.base"""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor."""
    return getattr(estimator, "_estimator_type", None) == "regressor"
