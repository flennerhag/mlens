"""Evaluator

Class for evaluating a set of estimators and preprocessing pipelines

Allows user to evaluate several models simoultanously across a set of pre-specified pipelines. The class is useful for comparing a set of estimators when several preprocessing pipelines have potential. By fitting all estimators on the same folds, number of fit can be greatly reduced as compared to pipelining each estimator and gitting them in an sklearn grid search. If preprocessing is time consuming, the evaluator class can be order of magnitued faster than a standard gridsearch.

If the user in unsure about what estimators to fit, the preprocess method can be used to preprocess data, after which the evuate method can be run any number of times upon the pre-made folds for various configurations of parameters. Current implementation only accepts randomized grid search.

### Parameters
``scoring`` : ``func``
    scoring function that follows sklearn API, i.e. ``score = scoring(estimator, X, y)``
``error_score`` : ``int``, score to assign when estimator fit fails
``preprocessing`` : ``dict``, default = ``None``
    dictionary of lists with preprocessing pipelines to fit models on. Each pipeline will be used to generate k folds that are stored, hence with large data running several cases with many cv folds can require considerable memory. preprocess should be of the form: ``P = {'case-1': [step1, step2], ...}``
``cv`` : ``int``, ``obj``, default = ``2``
    cross validation folds to use. Either pass a ``KFold`` class object that accepts as ``split`` method, or the number of folds to use in a standard ``KFold`` split
``shuffle`` : ``bool``, default = ``True``
    whether to shuffle data before creating folds
``random_state`` : ``int``, default = ``None``
    seed for creating folds (if ``shuffle`` = ``True``) and for parameter draws 
``n_jobs_preprocessing`` : ``int``, default = ``-1``
    number of CPU cores to use for preprocessing of folds
``n_jobs_estimators`` : ``int``, default = ``-1``
    number of CPU cores to use for grid search (estimator fitting)
``verbose`` : ``bool``, ``int``, default = ``False``
    level of printed output.

### Attributes

``summary_`` : ``DataFrame``
    Summary output that shows data for best mean test scores, such as test and train scores, std, fit times, and params
``cv_results_`` : ``DataFrame``
    a table of data from each fit. Includes mean and std of test and train scores and fit times, as well as param draw index and parameters.
``best_index`` : ``ndarray``,
    an array of index keys for best estimator in ``cv_results_``

### Methods

``preprocess`` : ``None``
    Method for preprocessing data separately from estimator evaluation. Helpful if preprocessing is costly relative to estimator fitting and flexibility is needed in evaluating estimators. Examples include fitting base estimators as part of preprocessing, to evaluate suitabe meta estimators in ensembles.

    **Parameters**
    ``X`` : array-like, shape = ``[n_samples, n_features]``
        input matrix to be used for prediction
    ``y`` : array-like, shape = ``[n_samples, ]``
        output vector to trained estimators on

    **Returns**
    ``dout`` : ``list``
        list of lists with folds data. For internal use.

``evaluate`` : ``estimators``, ``param_dicts``, ``n_iter``, ``reset_preprocess``, ``flush_preprocess``
    Method to run grid search on a set of estimators with given param_dicts for n_iter iterations. Set ``reset_preprocess`` to ``True`` to regenerate preprocessed data. Set ``flush_preprocess`` to drop data after evaluation.

    **Parameters**
    ``X`` : array-like, shape = ``[n_samples, n_features]``
        input matrix to be used for prediction
    ``y`` : array-like, shape = ``[n_samples, ]``
        output vector to trained estimators on
    ``estimators`` : ``dict``
        set of estimators to use: estimators={'est1': est(), ...}
    ``param_dicts`` : ``dict``
        param_dicts for estimators. Current implementation only supports randomized grid search, where passed distributions accept the ``.rvs()`` method. See ``sklearn.model_selection.RandomizedSearchCV`` for details.Form: ``param_dicts={'est1': {'param1': dist}, ...}``
    ``n_ier`` : ``int``
        number of parameter draws
    ``reset_preprocess`` : ``bool``, default = ``False``
        set to ``True`` to regenerate preprocessed folds
    ``flush_preprocess`` : ``bool``, default = ``False``
        set to ``True`` to drop preprocessed data. Useful if memory requirement is large.

    **Returns**
    ``self`` : ``obj``
        class instance with stored evaluation data
