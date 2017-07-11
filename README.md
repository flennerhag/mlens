![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
[![PyPI version](https://badge.fury.io/py/mlens.svg)](http://badge.fury.io/py/mlens)
[![Build Status](https://travis-ci.org/flennerhag/mlens.svg?branch=master)](https://travis-ci.org/flennerhag/mlens)
[![Build status](https://ci.appveyor.com/api/projects/status/99g65kvraic8w2la/branch/master?svg=true)](https://ci.appveyor.com/project/flennerhag/mlens/branch/master)
[![Code Health](https://landscape.io/github/flennerhag/mlens/master/landscape.svg?style=flat)](https://landscape.io/github/flennerhag/mlens/master)
[![Coverage Status](https://coveralls.io/repos/github/flennerhag/mlens/badge.svg?branch=master)](https://coveralls.io/github/flennerhag/mlens?branch=master)
[![Documentation Status](http://readthedocs.org/projects/mlens/badge/?version=stable)](http://mlens.readthedocs.io/en/stable/?badge=stable)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

# ML- Ensemble

**A Python library for memory efficient
parallelized ensemble learning**.

ML-Ensemble combines the Scikit-learn estimator API with a neural-network style
API to allow straight-forward multi-layer ensemble network architectures that
can be used as any Scikit-learn estimator.

The core principle of ML-Ensemble is to maximize parallel processing at minimum
memory consumption. ML-Ensemble uses memory mapping to remain memory neutral
during parallel estimation.

For full documentation, see [here](http://mlens.readthedocs.io/en/latest/).

## Core Features

### Transparent Architecture API

Ensembles are built by adding layers to an instance object: layers in their
turn are comprised of a list of estimators. No matter how complex the
ensemble, to train it call the ``fit`` method:

```Python
ensemble = Subsemble()

# First layer
ensemble.add(list_of_estimators)

# Second layer
ensemble.add(list_of_estimators)

# Final meta estimator
ensemble.add_meta(estimator)

# Train ensemble
ensemble.fit(X, y)
```

### Memory Efficient Parallelized Learning

Training data is persisted to a memory cache that each sub-process has access
to, allowing parallel processing to require no more memory than processing
on a single thread.
For more details, see [here](http://mlens.readthedocs.io/en/latest/memory.html).

Expect 95-97% of training time to be spent fitting the base estimators -
*irrespective* of data size. The time it takes to fit an ensemble depends
therefore entirely on how fast the chosen base learners are,
and how many CPU cores are available.

Moreover, ensemble classes that fit estimators on subsets scale more
efficiently than the base learners when these do not scale linearly.

### Modular build of multi-layered ensembles

The core unit of an ensemble is a **layer**, much as a hidden unit in a neural
network. Each layer contains an ensemble class specification and a mapping of
preprocessing pipelines to base learners to be used during fitting.

The modular build of an ensemble allows great flexibility in architecture,
both in terms of the depth of the ensemble (number of layers)
and how each layer generates predictions.

### Differentiated preprocessing pipelines

ML-Ensemble offers the possibility to specify, for each layer, a set
of preprocessing pipelines that maps to different (or the same) sets of
estimators. One set of estimators might benefit from one type of preprocessing,
while for a different set of estimators another preprocessing pipeline is
desired. This can easily be achieved in ML-Ensemble:

```Python

preprocessing = {'pipeline-1': list_of_transformers_1,
                 'pipeline-2': list_of_transformers_2}

estimators = {'pipeline-1': list_of_estimators_1,
              'pipeline-2': list_of_estimators_2}

ensemble.add(estimators, preprocessing)
```

### Dedicated Diagnostics

ML Ensemble implements a dedicated diagnostics and model selection suite
for intuitive and speedy ensemble evaluation. In particular, the ``Evaluator``
class allows users to evaluate several preprocessing pipelines and several
estimators in one go, possibly even evaluating different estimators on
different preprocessing pipelines. 

Moreover, the ``EnsembleTransformer`` allows user to build a transformer that
behaves as any ensemble, but that returns the predictions from the ``fit``
call. Hence, the transformer can be used together with the ``Evaluator`` to
pre-generate training folds for model selection of the meta estimator
(or further layers). For a full example, see [here](http://mlens.readthedocs.io/en/latest/ensemble_tutorial.html#meta-learner-model-selection).

```Python

preprocessing_dict = {'pipeline-1': list_of_transformers_1,
                      'pipeline-2': list_of_transformers_2}


evaluator = Evaluator(scorer=score_func)
evaluator.fit(X, y, list_of_estimators, param_dists_dict, preprocessing_dict)
```

Output is ordered per transformation case and estimators, and can be given a
readable tabular view by passing the ``summary`` attribute to a Pandas
``DataFrame``, as in the example below.

```Python
              fit_time_mean  fit_time_std  train_score_mean  train_score_std  test_score_mean  test_score_std               params
prep-1 est-1       0.001353      0.001316          0.957037         0.005543         0.960000        0.032660                   {}
       est-2       0.000447      0.000012          0.980000         0.004743         0.966667        0.033333  {'n_neighbors': 15}
prep-2 est-1       0.001000      0.000603          0.957037         0.005543         0.960000        0.032660                   {}
       est-2       0.000448      0.000036          0.965185         0.003395         0.960000        0.044222   {'n_neighbors': 8}
prep-3 est-1       0.000735      0.000248          0.791111         0.019821         0.780000        0.133500                   {}
       est-2       0.000462      0.000143          0.837037         0.014815         0.800000        0.126491   {'n_neighbors': 9}
```

## How to install

#### PyPI

Execute

```bash
pip install mlens
```

#### Bleeding edge

To ensure latest version is installed, fork the GitHub repository.

```bash
git clone https://flennerhag/mlens.git; cd mlens;
python install setup.py
```

#### Developer version

For the latest in-development version, install the ``dev`` branch of the
repository.

```bash
git clone https://flennerhag/mlens.git; cd mlens; git checkout dev;
python install setup.py
```

## License

MIT License

Copyright (c) 2017 Sebastian Flennerhag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
