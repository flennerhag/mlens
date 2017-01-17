#ML-Ensemble

*ML Ensemble is a Python library for building fully parallelized ensembles with a Scikit-learn's API. It is fully compatible with Scikic-learn objects such as pipelines and grid search classes.**

The package's ensemble classes are built to allow a highly flexible ensemble architecture, while easily integrated in a full stack prediction pipeline.
A key feature of the ML-Ensemble is the use of *base pipelines* as inputs to the the basic ensemble class. This allows the user to specify which base estimators share a common preprocessing pipeline, to avoid preprocessing the data for each estimator. This can dramatically increase speed if 
the dataset is large, preprocessing is heavy, or an expansive grid search needs to be perfomed. 

The backend of ML-Ensemble is built on [*joblibb*](https://pythonhosted.org/joblib/parallel.html) to maximize the parallelization of fitting and training base esimtators. Hence, ML-Ensemble is optimized for speed. 

The project is currently in development, and currently the following front-end classes are implemented:
- `StackingEnsemble`: a one-stop-shop for generating and training ensembles. See [here](mlens/examples/example.ipynb) for an example.
- `PredictionFeature`: an sklearn compatibile class for generating a feature of out-of-sample predicitons. In pipeline, coming soon.
- `Evaluator`: a one-stop-shop for model evaluation that allows you to compare in one table the performance of any number of models, across any number of preprocessing pipelines. By fitting all estimators in one go, grid search time is dramatically reduced as compared to grid search one pipelined model at a time. See [here](mlens/test/example_evaluator.ipynb) for an example.

The current implementation foces on moderately sized datasets and achieves maximium parallelization by pre-making preprocessing pipelines and temporarily storing pre-made data during the ``fit`` call. With very large datasets, pre-making folds 
may be overly costly from a memory point of view. A future version will allow the user to instead cache fitted estimator pipelines: this will avoid creating any excess data and minimize number of transformer ``fit`` calls, but will require a ``transform`` call for each estimator fitting.


If you'd like to contribute, don't hesitate to reach out!

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
