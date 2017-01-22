# Stacking Ensemble

Meta estimator class that blends a set of base estimators via a meta estimator. In difference to standard stacking, where the base estimators predict the same data they were fitted on, this class uses k-fold splits of the the training data make base estimators predict out-of-sample training data. Since base estimators predict training data as in-sample, and test data as out-of-sample, standard stacking suffers from a bias in that the meta estimators fits based on base estimator training error, but predicts based on base estimator test error. This blends overcomes this by splitting up the training set in the fitting stage, to create near identical for both training and test set. Thus, as the number of folds is increased, the training set grows closer in remeblance of the test set, but at the cost of increased fitting time.

## Parameters

``meta_estimator`` : ``obj``
    estimator to fit on base_estimator predictions. Must accept fit and predict method.

``base_pipelines`` : ``dict``, ``list``
    base estimator pipelines. If no preprocessing, pass a list of estimators, possible as named tuples ``[('est-1', est), (...)]``. If preprocessing is desired, pass a dictionary with pipeline keys: ``{'pipe-1': [preprocessing], [estimators]}``, where ``[preprocessing]`` should be a list of transformers, possible as named tuples, and estimators should be a list of estimators to fit on preprocesssed data, possibly as named tuples. General format should be ``{'pipe-1', [('step-1', trans), (...)], [('est-1', est), (...)]}``, where named each step is optional. Each transformation step and estimators must accept fit and transform / predict methods respectively

``folds`` : ``int``, ``obj``, default = ``2``
    number of folds to use for constructing meta estimator training set. Either pass a KFold class object that accepts as ``split`` method, or the number of folds in standard KFold

``shuffle`` : ``bool``, default = ``True``
    whether to shuffle data for creating k-fold out of sample predictions

``as_df`` : ``bool``, default = ``False``
    whether to fit meta_estimator on a dataframe. Useful if meta estimator allows feature importance analysis

``scorer`` : ``func``, default = ``None``
    scoring function. If a function is provided, base estimators will be scored on the training set assembled for fitting the meta estimator. Since those predictions are out-of-sample, the scores represent valid test scores. The scorer should be a function that accepts an array of true values and an array of predictions: ``score = f(y_true, y_pred)``. The scoring function of an sklearn scorer can be retrieved by ``._score_func``

``random_state`` : ``int``, default = ``None``
    seed for creating folds during fitting

``verbose`` : ``bool``, ``int``, default = ``False``
    level of verbosity of fitting:

    - ``verbose = 0`` prints minimum output
    - ``verbose = 1`` give prints for meta and base estimators
    - ``verbose = 2`` prints also for each stage (preprocessing, estimator)
    - ``n_jobs`` : ``int``, default = ``-1`` number of CPU cores to use for fitting and prediction

## Attributes

``scores_`` : ``dict``
    scored base of base estimators on the training set, estimators are named according as pipeline-estimator.

``base_estimators_`` : ``list``
    fitted base estimators

``base_columns_`` : ``list``
    ordered list of base estimators as they appear in the input matrix to the meta estimators. Useful for mapping sklearn feature importances, which comes as ordered ``ndarrays``.

``preprocess_`` : ``dict``
    fitted preprocessing pipelines

## Methods

``fit`` : ``X``, ``y`` = ``None``
    Method for fitting ensemble

    **Parameters**

    ``X`` : array-like, shape = ``[n_samples, n_features]``
        input matrix to be used for prediction

    ``y`` : array-like, shape = ``[n_samples, ]``
        output vector to trained estimators on

    **Returns**

    ``self`` : ``obj``
        class instance with fitted estimators

``predict`` : ``X``
    Predict with fitted ensemble

    **Parameters**

    ``X`` : array-like, shape = ``[n_samples, n_features]``
        input matrix to be used for prediction

    **Returns**
    ``y`` : array-like, shape = ``[n_samples, ]``
        predictions for provided input array

``get_params`` : ``None``
    Method for generating mapping of parameters. Sklearn API.


# Evaluator

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
