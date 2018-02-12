"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Classes for implementing various cross-validation strategies. By default,
ML-Ensemble indexers generates list of tuples, as opposed to array indexes,
to avoid serialization during multiprocessing.
"""

from .base import FullIndex, BaseIndex, prune_train, make_tuple, partition
from .fold import FoldIndex
from .blend import BlendIndex
from .subsemble import SubsetIndex, ClusteredSubsetIndex


INDEXERS = {'stack': FoldIndex,
            'blend': BlendIndex,
            'subsemble': SubsetIndex,
            'clusteredsubsemble': ClusteredSubsetIndex,
            'full': FullIndex
            }


__all__ = ['BaseIndex',
           'BlendIndex',
           'FoldIndex',
           'SubsetIndex',
           'FullIndex',
           'ClusteredSubsetIndex',
           'prune_train',
           'partition',
           'make_tuple'
           ]
