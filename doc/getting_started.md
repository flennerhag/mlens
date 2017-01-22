# Getting Started


The library utilizes the Scikit-learn API. Specify a set of base estimators, either as a list of estimators or a dictionary of preprocessing cases with associated base estimators, along with a meta estimator. In the simplest case: 

```Python
from mlens.ensemble import StackingEnsemble
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.datasets import load_boston

# Some data
X, y = load_boston(return_X_y=True)

# Base estimators and meta estimator
base = [SVR(), RandomForest()]
meta = Lasso()

# Ensemble
ens = StackingEnsemble(meta, base)

ens.fit(X, y)
predictions = ens.predict(X)
```

The user can specify *different preprocessing pipelines* for subsets of base estimators, thereby minimizing the transformation calls. If two base estimators require one type of preprocessing, and two other require a different preprocessing pipeline, typically the user would have to specify a preprocessing pipeline for each estimator despite several of them sharing the same pipeline. ML-Ensemble solves this problem by allowing the user to specify a straightforward mapping of preprocessing pipelines and base estimators. Therefore preprocessing is done once with relevant estimators fitted on the preprocessed data. 

To differentiate preprocessing, create a mapping between preprocessing pipelines (which is an ordered list of ``transformers`` that follows the Scikit-learn API (accepts a ``fit`` and ``transform`` call)) and a list of base estimators. Suppose that we wanted the training set for the SVR and the Random forest to be min-max scaled, and the training set for the others to be standardized. We can easliy achieved this by passing a dictionary of base estimator piplines:

```python
ensemble = StackingEnsemble(meta_estimator=LinearRegression(),
# a pipeline is a dict of tuples
# of the form (preprocess, estimators),
# where preprocess is a list of transformers, and
# estimators is a list of base estimators
base_pipelines={'sc': ([StandardScaler()],
[Lasso(), SVR()]),
'mm': ([MinMaxScaler()],
[KNeighborsRegressor(),
RandomForestRegressor()])

```

In this way, ML-Ensemble can build flexible ensemble architectures through an easy user interface and allows maximally parallelized fitting of the ensemble. No need to write hundreds or even thousands of lines of code to build your ensemble, ML-Ensemble has alreay done if for you! 

For an example notebook, see the repository's [examples](https://github.com/flennerhag/mlens/tree/master/doc/examples).
