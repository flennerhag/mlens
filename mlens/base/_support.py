#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 22/01/2017
licence: MIT
Support function for ensemble classes
"""


def _check_estimators(full_fit_est, fold_fit_est, layer):
    """Helper function to check that fitted estimators overlap"""
    if not all([est in full_fit_est for est in fold_fit_est]):
        raise ValueError('[%s] Not all estimators successfully fitted on the full '
                         'dataset were fitted during fold predictions. Aborting.'
                         '\n[%s] Fitted estimators on full data: %r'
                         '\n[%s] Fitted estimators on folds: %r' % (layer, layer, full_fit_est, layer, fold_fit_est))

    if not all([est in fold_fit_est for est in full_fit_est]):
        raise ValueError('[%s] Not all estimators successfully fitted on the fold '
                         'data were successfully fitted on the full data. Aborting.'
                         '\n[%s] Fitted estimators on full data: %r'
                         '\n[%s] Fitted estimators on folds: %r' % (layer, layer, full_fit_est, layer, fold_fit_est))


def _name_columns(estimator_cases):
    """Utility func for naming a mapping of estimators on different cases"""
    return [case + '-' + est_name if case not in [None, ''] else est_name
            for case, estimators in estimator_cases.items()
            for est_name, _ in estimators]
