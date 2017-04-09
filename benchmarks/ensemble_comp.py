"""ML-ENSEMBLE

Comparison of ensemble performance across scale.
"""

import numpy as np

from mlens.ensemble import BlendEnsemble, SuperLearner, Subsemble

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from time import perf_counter

def get_data(d):
    """Return train and test set."""
    data = np.DataSource()
    if d == 'train':
        out = data.open('http://archive.ics.uci.edu/ml/'
                        'machine-learning-databases/'
                        'poker/poker-hand-training-true.data')
    elif d == 'test':
        out = data.open('http://archive.ics.uci.edu/ml/'
                        'machine-learning-databases/'
                        'poker/poker-hand-testing.data')
    else:
        raise ValueError("Not valid data option.")

    X = np.loadtxt(out, delimiter=",")
    y = X[:, -1]
    X = X[:, :-1]
    return X, y

xtrain, ytrain = get_data('train')
xtest, ytest = get_data('test')

estimators = {'subsemble': Subsemble(),
              'super_learner': SuperLearner(),
              'blend_ensemble': BlendEnsemble()}

base_learners = [RandomForestClassifier(n_estimators=500,
                                        max_depth=10,
                                        min_samples_split=50,
                                        max_features=0.6),
                 LogisticRegression(C=1e5),
                 GradientBoostingClassifier()]

for clf in estimators.values():
    clf.add([RandomForestClassifier(), LogisticRegression(), MLPClassifier()])
    clf.add_meta(SVC())


times, scores = {k: [] for k in estimators}, {k: [] for k in estimators}
for i in range(5000, xtrain.shape[0] + 5000, 5000):
    for name, clf in estimators.items():

        t0 = perf_counter()
        clf.fit(xtrain[:i], ytrain[:i])
        times[name].append(perf_counter() - t0)

        scores[name].append(accuracy_score(ytest, clf.predict(xtest)))

        print('{:5} | {:20} | {:2.3f} | {:2.3f}'.format(i,
                                                          name,
                                                          scores[name][-1],
                                                          times[name][-1]))
