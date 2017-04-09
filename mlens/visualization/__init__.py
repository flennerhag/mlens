"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT
"""

import warnings

try:
    from seaborn import set_palette
    set_palette('husl', 100)
except ImportError:
    pass

try:
    from .correlations import corrmat, clustered_corrmap, corr_X_y
    from .var_analysis import pca_comp_plot, pca_plot, exp_var_plot
except ImportError:
    warnings.warn("Matplotlib and Seaborn not installed. Cannot load "
                  "visualization module.", ImportWarning)
    corrmat = None
    clustered_corrmap = None
    corr_X_y = None
    pca_plot = None
    pca_comp_plot = None
    exp_var_plot = None

__all__ = ['corrmat', 'clustered_corrmap', 'corr_X_y',
           'pca_comp_plot', 'pca_plot', 'exp_var_plot']
