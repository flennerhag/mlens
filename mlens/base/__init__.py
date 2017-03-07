"""ML-ENSEMBLE

author: Sebastian Flennerhag
licence: MIT
Base modules used throughout mlens
"""

from .clone import clone_base_estimators, clone_preprocess_cases
from .support import check_fit_overlap, name_columns, safe_slice, \
    name_estimators, name_layer, check_instances
from .id_train import IdTrain

__all__ = ['clone_base_estimators', 'clone_preprocess_cases',
           'check_fit_overlap', 'name_columns', 'safe_slice',
           'name_estimators', 'name_layer',  'check_instances',
           'IdTrain']
