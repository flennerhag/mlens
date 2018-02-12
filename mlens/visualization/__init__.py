"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT
"""

try:
    from seaborn import set_palette
    set_palette('husl', 100)
except ImportError:
    pass

from .correlations import corrmat, clustered_corrmap, corr_X_y
from .var_analysis import pca_comp_plot, pca_plot, exp_var_plot

__all__ = ['corrmat', 'clustered_corrmap', 'corr_X_y',
           'pca_comp_plot', 'pca_plot', 'exp_var_plot']
