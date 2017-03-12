"""
:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT
"""

import warnings

try:
    from .correlations import corrmat, clustered_corrmap, corr_X_y
    from .var_analysis import IO_plot_comp, IO_plot, exp_var_plot
except ImportError:
    warnings.warn("Matplotlib and Seaborn not installed. Cannot load "
                  "visualization module.", ImportWarning)
    corrmat = None
    clustered_corrmap = None
    corr_X_y = None
    IO_plot = None
    IO_plot_comp = None
    exp_var_plot = None

__all__ = ['corrmat', 'clustered_corrmap', 'corr_X_y',
           'IO_plot_comp', 'IO_plot', 'exp_var_plot']
