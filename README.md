![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
[![Build Status](https://travis-ci.org/flennerhag/mlens.svg?branch=master)](https://travis-ci.org/flennerhag/mlens)
[![Coverage Status](https://coveralls.io/repos/github/flennerhag/mlens/badge.svg?branch=master)](https://coveralls.io/github/flennerhag/mlens?branch=master)
[![PyPI version](https://badge.fury.io/py/mlens.svg)](http://badge.fury.io/py/mlens)
[![Code Health](https://landscape.io/github/flennerhag/mlens/master/landscape.svg?style=flat)](https://landscape.io/github/flennerhag/mlens/master)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

# ML Ensemble - A library for ensemble learning

**ML Ensemble is a Python library for building [stacked ensembles](http://mlwave.com/kaggle-ensembling-guide/) through Scikit-learn's API to allow for full parallelization and integration with sklearn's cross-validation, pipelining and diagnostics tools.**


The library comes with:
- `Ensemble`: a one-stop-shop for generating and training ensembles. See [here](mlens/test/example.ipynb) for an example.
- `PredictionFeature`: an sklearn compatibile class for generating a feature of out-of-sample predicitons. Coming soon.
- `Evaluator`: a one-stop-shop for model evaluation that allows you to compare in one table the performance of any number of models, across any number of preprocessing pipelines. By fitting all estimators in one go, grid search time is dramatically reduced as compared to grid search one pipelined model at a time. See [here](mlens/test/example_evaluator.ipynb) for an example.

## How to install

#### PyPI

Execute  

```bash
pip install mlens  
```

#### Bleeding edge

To ensure latest version is installed, fork the GitHub repository and install mlxtens using the symlink options.

```bash
git clone https://flennerhag/mlens.git; cd mlens;
pip install -e .
```

To update the package, pull the latest changes from the github repo

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
