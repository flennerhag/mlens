"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Testing objects
"""
from .dummy import (Data,
                    EstimatorContainer,
                    run_learner,
                    get_learner,
                    get_layer,
                    run_layer,
                    InitMixin,
                    )

__all__ = ['Data',
           'EstimatorContainer',
           'get_learner',
           'run_learner',
           'get_layer',
           'run_layer',
           'InitMixin'
           ]
