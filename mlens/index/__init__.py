"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Base modules
"""

from .indexer import (FoldIndex,
                      BlendIndex,
                      SubsetIndex,
                      FullIndex,
                      ClusteredSubsetIndex)


INDEXERS = {'stack': FoldIndex,
            'blend': BlendIndex,
            'subsemble': SubsetIndex,
            'clusteredsubsemble': ClusteredSubsetIndex,
            'full': FullIndex
            }


__all__ = ['BlendIndex',
           'FoldIndex',
           'SubsetIndex',
           'FullIndex',
           'ClusteredSubsetIndex'
           ]
