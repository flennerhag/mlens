#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
Support functions for naming estimators to enable ensemble parameter mapping
"""

from __future__ import division, print_function
from sklearn.pipeline import Pipeline, _name_estimators


def name_estimators(estimators, prefix='', suffix=''):
    """Function for creating dict with named estimators for get_params"""
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
    """Function for naming estimators and transformers for parameter setting"""
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


def check_estimators(estimators):
    """Remove potential names from passed list of estimators"""
    if len(estimators) == 0:
        return []
    else:
        if isinstance(estimators[0], tuple):
            estimators = [est for _, est in estimators]
        return estimators


def _check_names(estimators):
    """Helper to ensure all estimators and transformers are named"""
    # Check if empty
    if len(estimators) is 0:
        return []
    # Check if pipeline, if so split up
    elif isinstance(estimators, Pipeline):
        return estimators.steps
    # Check if named tuple
    elif isinstance(estimators[0], tuple):
        return estimators
    # Else assume list of unnamed estimators
    else:
        return _name_estimators(estimators)


def _split_base(base_pipelines):
    """Utility function for splitting the base pipeline"""
    # if preprocessing pipes, seperate base estimators and preprocessing
    if isinstance(base_pipelines, dict):
        preprocess = [(case, _check_names(p[0])) for case, p in
                      base_pipelines.items()]
        base_estimators = [(case, _check_names(p[1])) for case, p in
                           base_pipelines.items()]
    # else, ensure base_estimators are named
    else:
        preprocess = []
        base_estimators = [('', _check_names(base_pipelines))]
    return preprocess, base_estimators
