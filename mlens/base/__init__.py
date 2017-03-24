"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base modules
"""

from .id_train import IdTrain
from .indexer import FoldIndex, BlendIndex, SubSampleIndexer, FullIndex


INDEXERS = {'stack': FoldIndex,
            'blend': BlendIndex,
            'subset': SubSampleIndexer,
            'full': FullIndex}


__all__ = ['IdTrain', 'BlendIndex', 'FoldIndex', 'SubSampleIndexer',
           'FullIndex']
