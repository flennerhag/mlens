"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Graph handles for deep computational graphs and ready-made ensemble classes
for ensemble networks. Ready-made classes are full Scikit-learn estimators and
can be used in conjunction with any other standard estimator.
"""

from .super_learner import SuperLearner
from .blend import BlendEnsemble
from .subsemble import Subsemble
from .sequential import SequentialEnsemble
from .base import Sequential, BaseEnsemble

__all__ = ['SuperLearner',
           'BlendEnsemble',
           'Subsemble',
           'SequentialEnsemble',
           'Sequential',
           'BaseEnsemble']
