#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 11/01/2017
licence: MIT
Base modules for mlens
"""
from ._clone import _clone_base_estimators, _clone_preprocess_cases
from ._setup import (name_estimators, name_base, check_estimators,
                     _check_names, _split_base)
from ._support import _check_estimators, name_columns

__all__ = ['_clone_base_estimators', '_clone_preprocess_cases',
           'name_estimators', 'name_base', 'check_estimators', '_check_names',
           '_split_base', '_check_estimators', 'name_columns']
