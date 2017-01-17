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

For example notebooks, see the repository's [examples](https://github.com/flennerhag/mlens/tree/master/mlens/examples).